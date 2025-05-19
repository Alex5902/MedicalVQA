from collections import defaultdict, OrderedDict
import torch.nn.functional as F
from tqdm.auto import tqdm
from copy import deepcopy
from time import time
from torch import nn
import torch
import pdb
import sys

from utils import get_merging_fn, get_mask_fn
from masking_ops import masked_merge


class VectorOps(nn.Module):
    def directions_to_reps(self, directions):
        if isinstance(directions, list):
            return [self.directions_to_reps(direction) for direction in directions]
        return torch.nn.utils.parameters_to_vector(
            [value.reshape(-1) for key, value in directions.items()]
        )
        
    def rep_to_state_dict(self, vector, state_dict, remove_keys=[]):
        if isinstance(vector, list) or len(vector.shape) == 2:
            return [self.rep_to_state_dict(v, state_dict, remove_keys) for v in vector]
        # create a reference dict to define the order of the vector
        reference_dict = deepcopy(state_dict)
        for key in remove_keys:
            if key in reference_dict:
                del reference_dict[key]
        sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

        # create a shared state dict using the refence dict
        torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

        # add back the encoder and decoder embedding weights.
        if "transformer.shared.weight" in sorted_reference_dict:
            for key in remove_keys:
                sorted_reference_dict[key] = sorted_reference_dict[
                    "transformer.shared.weight"
                ]
        return sorted_reference_dict
    
    def mask_to_state_dict(self, mask, state_dict, remove_keys=[]):
        if isinstance(mask, list):
            return [self.mask_to_state_dict(m, state_dict, remove_keys) for m in mask]
        return self.rep_to_state_dict(mask, state_dict, remove_keys)
    
    def forward(self, directions, merging_fn, merge_config):
        vectors = self.directions_to_reps(directions)
        merged_vector,rows_to_keep, topk_mask = merging_fn(vectors)
        mask_sd = self.rep_to_state_dict(topk_mask, directions[0])
        
        ties_mask = [dict() for _ in range(len(rows_to_keep))]
        for idx in range(len(rows_to_keep)):
            ties_mask[idx] = self.rep_to_state_dict(rows_to_keep[idx], directions[0])
        sd = self.rep_to_state_dict(merged_vector, directions[0])
        
        return sd, ties_mask


class TaskMerger(nn.Module):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__()
        
        self.device = device
        # Ensure finetuned_models is a list for len()
        if not isinstance(finetuned_models, list):
            finetuned_models = [finetuned_models] if finetuned_models is not None else []
            
        self.scaling_coeffs = torch.tensor([1.] * len(finetuned_models) if finetuned_models else [1.])
        self.param_handler = param_handler
        self.finetuned_models = finetuned_models
        
        # Handle case where finetuned_models might be empty or None
        if self.finetuned_models and all(self.finetuned_models):
            self.ftms_params = [param_handler(ft_model).get_ft_parameters() for ft_model in finetuned_models]
        else:
            self.ftms_params = []
            print("Warning TaskMerger: finetuned_models list is empty or contains None. ftms_params will be empty.")

        if pretrained_model is not None:
            # If pretrained_model is a state_dict already (like our dummy zeros_sd)
            if isinstance(pretrained_model, dict):
                 self.pt_params = pretrained_model 
            # If it's a model instance or a handler instance
            elif hasattr(pretrained_model, 'state_dict') and callable(pretrained_model.state_dict):
                 self.pt_params = pretrained_model.state_dict()
            elif hasattr(pretrained_model, 'get_ft_parameters') and callable(pretrained_model.get_ft_parameters): # If it's a handler
                 self.pt_params = pretrained_model.get_ft_parameters()
            else:
                raise ValueError("pretrained_model in TaskMerger is not a model, state_dict, or recognized handler.")
            # No need to move to CPU if it's already a CPU state_dict from LoRAHandler
        else:
            self.pt_params = {} # or handle as an error if always required
            print("Warning TaskMerger: pretrained_model is None. pt_params will be empty.")
            
        self.merge_config = merge_config if merge_config is not None else {}


    def randbin(self, M, N, P):
        P = 1-P
        return torch.randint(2, size=(M, N), dtype=torch.float32, device=self.device).bernoulli(P) # Added device
    
    def apply_dare(self, ftms_params_list_of_sds, p, dare_seed = 0):
        # This method expects ftms_params to be a list of state_dicts (DeltaWs)
        print("DARE seed: ", dare_seed)
        torch.manual_seed(dare_seed)
        dared_directions = []
        for ftm_sd in ftms_params_list_of_sds:
            direction_sd = OrderedDict() # Keep order
            for key, finetuned_val_delta_w in ftm_sd.items():
                # Ensure val is on the correct device for randbin
                val_on_device = finetuned_val_delta_w.to(self.device)
                if val_on_device.ndim < 2: # Ensure val is at least 2D for randbin M,N
                    print(f"Warning DARE: Skipping key {key} due to insufficient dimensions ({val_on_device.ndim}) for randbin.")
                    direction_sd[key] = val_on_device # Keep original if cannot apply DARE
                    continue
                try:
                    direction_sd[key] = val_on_device * self.randbin(val_on_device.shape[0], val_on_device.shape[1], p) * (1/(1-p))
                except IndexError as e: # Catches if shape is (X,) making shape[1] fail
                    print(f"Warning DARE: Error applying randbin for key {key} with shape {val_on_device.shape}. Error: {e}. Keeping original.", file=sys.stderr)
                    direction_sd[key] = val_on_device
            dared_directions.append(direction_sd)
        return dared_directions

    def get_task_directions(self, ptm_params_sd, ftms_params_list_of_sds):
        # This method calculates DeltaW = FT - PTM
        # In our case, ftms_params_list_of_sds are already DeltaWs from LoRAHandler,
        # and ptm_params_sd is a dict of zeros. So, this should return DeltaWs.
        finetuned_directions = []
        for ftm_sd in ftms_params_list_of_sds: # ftm_sd is one task's DeltaW state_dict
            direction_sd = OrderedDict()
            for key, finetuned_val_delta_w in ftm_sd.items():
                # ptm_val should be zero for the corresponding key
                ptm_val = ptm_params_sd.get(key, torch.zeros_like(finetuned_val_delta_w))
                direction_sd[key] = finetuned_val_delta_w - ptm_val # Should be DeltaW - 0 = DeltaW
            finetuned_directions.append(direction_sd)
        return finetuned_directions
    
    def set_scaling_coeffs(self, scaling_coeffs):
        num_models = len(self.ftms_params) if self.ftms_params else 1
        if isinstance(scaling_coeffs, float) or (isinstance(scaling_coeffs, list) and len(scaling_coeffs) == 1):
            coeff_val = scaling_coeffs if isinstance(scaling_coeffs, float) else scaling_coeffs[0]
            self.scaling_coeffs = torch.tensor([coeff_val] * num_models)
        elif isinstance(scaling_coeffs, list) and len(scaling_coeffs) == num_models:
            self.scaling_coeffs = torch.tensor(scaling_coeffs)
        else:
            print(f"Warning: scaling_coeffs length mismatch or type error. Expected float or list of length {num_models}. Got {scaling_coeffs}. Defaulting to ones.")
            self.scaling_coeffs = torch.tensor([1.] * num_models)

    def get_layer_names(self, state_dict_of_delta_w):
        """
        Identifies layers from a state_dict. For DeltaW from LoRAHandler,
        keys are base module paths and should be treated as 'weight' matrices.
        """
        layer_names = defaultdict(lambda: dict())
        for key in state_dict_of_delta_w.keys():
            # Keys from LoRAHandler's get_ft_parameters are like 'bert.encoder.layer.0.attention.self.query'
            # These represent the DeltaW matrices themselves.
            # For SVDMerger's directions_to_matrices, these should be identified as the 'weight' part.
            if not ('.bias' in key or '_bias' in key): # LoRA DeltaW usually doesn't have explicit bias terms
                layer_names[key]['weight'] = key # The key itself is the identifier for the main matrix
            # If LoRAHandler ever started returning biases for LoRA layers (uncommon):
            # elif ('.bias' in key) or ('_bias' in key):
            #     strip_key = key.replace('.bias', '').replace('_bias', '')
            #     layer_names[strip_key]['bias'] = key
            else:
                # This case should ideally not be hit if input is purely DeltaW from LoRAHandler
                print(f"Warning get_layer_names: Key '{key}' not classified as weight or bias, treating as other.")
                layer_names[key]['other'] = key + ':other' # Fallback
        return layer_names
    
    def add_task_parameters(self, base_model_instance, parameters_sd, concat_across_output=True, scaling_coeffs=1.):
        # This method adds the merged DeltaW (parameters_sd) to a base model instance.
        # For LoRA, we don't add to a base model; we create a new LoRA adapter.
        # This method is more for full fine-tuning merging.
        # In our LoRA merging script, we'll handle saving the merged A/B matrices differently.
        print("Info TaskMerger.add_task_parameters: This method is for adding merged DeltaW to a base model. For LoRA, you'll typically factorize DeltaW and create a new adapter.")
        
        # The original logic (if we were to modify a base model directly):
        # if isinstance(parameters_sd, list): # Should not be a list for final merged parameters
        #     raise ValueError("add_task_parameters expects a single state_dict for merged parameters, not a list.")

        current_base_sd = base_model_instance.state_dict()
        with torch.no_grad():
            for key, val_delta_w in parameters_sd.items():
                if key in current_base_sd:
                    # Ensure val_delta_w is on the same device and dtype as current_base_sd[key]
                    target_device = current_base_sd[key].device
                    target_dtype = current_base_sd[key].dtype
                    
                    val_to_add = val_delta_w.to(target_device, dtype=target_dtype) * scaling_coeffs
                    
                    if not concat_across_output: # KnOTS specific for some models
                        val_to_add = val_to_add.T
                    
                    current_base_sd[key].add_(val_to_add)
                else:
                    print(f"Warning add_task_parameters: Key '{key}' from merged parameters not found in base_model. Skipping.")
        base_model_instance.load_state_dict(current_base_sd)
        return base_model_instance
    
    def directions_to_matrices(self, directions_list_of_sds, reference_layer_names_map=None):
        """
        Converts a list of state_dicts (each being a task's DeltaW)
        into a list of dictionaries, where each inner dict maps a layer_base_name
        to its 2D matrix representation.
        """
        all_matrices_for_all_tasks = []
        
        # Determine reference_layer_names_map from the first task's DeltaW if not provided
        # This ensures all tasks are processed based on the same layer structure.
        if not directions_list_of_sds:
            return []
        if reference_layer_names_map is None:
            reference_layer_names_map = self.get_layer_names(directions_list_of_sds[0])

        print(f"DEBUG directions_to_matrices: reference_layer_names_map (first 5 items): {list(reference_layer_names_map.items())[:5]}")

        for task_idx, task_delta_w_sd in enumerate(directions_list_of_sds):
            matrices_for_current_task = {}
            for layer_base_name, param_type_map in reference_layer_names_map.items():
                # layer_base_name is e.g., 'bert.encoder.layer.0.attention.self.query'
                # param_type_map is e.g., {'weight': 'bert.encoder.layer.0.attention.self.query'}
                
                if 'weight' in param_type_map:
                    weight_key_in_sd = param_type_map['weight'] # This will be same as layer_base_name
                    
                    if weight_key_in_sd not in task_delta_w_sd:
                        print(f"Warning directions_to_matrices: Layer '{weight_key_in_sd}' not found in task {task_idx} DeltaW. Skipping.", file=sys.stderr)
                        # Potentially fill with zeros of expected shape if strict layer matching is needed by SVDMerger
                        # For now, skip, SVDMerger's dict_of_concat_matrices will handle missing keys if any.
                        continue

                    delta_w_matrix = task_delta_w_sd[weight_key_in_sd] # This is the (Out, In) matrix
                    
                    # KnOTS' SVDMerger flattens the weight matrix for some reason before concatenation.
                    # Original: matrices[layer_name] = weight.flatten(1)
                    # If weight is already 2D (Out, In), flatten(1) is a no-op if In > 1.
                    # If In == 1, it makes it (Out,). SVD needs 2D.
                    # Let's ensure it's at least 2D.
                    if delta_w_matrix.ndim == 1: # Should not happen for LoRAHandler output
                        print(f"Warning directions_to_matrices: DeltaW for {weight_key_in_sd} is 1D. Unsqueezing.", file=sys.stderr)
                        matrix_to_store = delta_w_matrix.unsqueeze(0) # Or unsqueeze(1) depending on convention
                    elif delta_w_matrix.ndim == 0: # Scalar, highly unlikely
                         print(f"Warning directions_to_matrices: DeltaW for {weight_key_in_sd} is 0D. Skipping.", file=sys.stderr)
                         continue
                    else: # Already 2D or more
                        matrix_to_store = delta_w_matrix
                    
                    # The SVD part in KnOTS seems to handle matrices of original W_out x W_in.
                    # No further flattening seems necessary if LoRAHandler gives (O,I) matrices.
                    # The original `weight.flatten(1)` in KnOTS might have been for ViT patch embeddings etc.
                    matrices_for_current_task[layer_base_name] = matrix_to_store

                    # Original KnOTS code handles bias by concatenating it:
                    # if 'bias' in param_type_map:
                    #     bias = directions_sd[param_type_map['bias']]
                    #     matrices_for_current_task[layer_base_name] = torch.concat(
                    #         (matrices_for_current_task[layer_base_name], bias.reshape(-1, 1)), dim=1
                    #     )
                    # For LoRA DeltaW, there's typically no separate bias from LoRAHandler.

                elif 'other' in param_type_map: # Should not be hit if input is purely DeltaW from LoRAHandler
                    other_key_in_sd = param_type_map['other'].replace(':other', '')
                    if other_key_in_sd in task_delta_w_sd:
                        other_parameter = task_delta_w_sd[other_key_in_sd].to(torch.float32)
                        if len(other_parameter.shape) == 1: other_parameter = other_parameter[None, :]
                        elif len(other_parameter.shape) > 2: other_parameter = other_parameter.flatten(1)
                        matrices_for_current_task[layer_base_name + ':other'] = other_parameter
            all_matrices_for_all_tasks.append(matrices_for_current_task)
        
        # SVDMerger.transform calls this method with ftms_task_dirs (list of SDs)
        # and expects a list of dicts of matrices back.
        return all_matrices_for_all_tasks
    
    def matrix_to_state_dict(self, matrix_sd, reference_original_sd, remove_keys=[]):
        """
        Converts a state_dict of matrices (layer_base_name -> merged_matrix)
        back to a standard parameter state_dict (param_name -> tensor).
        reference_original_sd is a state_dict from one of the original DeltaWs (e.g., task_qformer_delta_w_sds[0])
        to get original shapes and dtypes.
        """
        if isinstance(matrix_sd, list): # Should not be a list for the final merged matrix_sd
            raise ValueError("matrix_to_state_dict expects a single dict of merged matrices.")

        # Use the structure of the original DeltaW state_dict for names and shapes
        # The keys in matrix_sd are layer_base_names like 'bert.encoder.layer.0.attention.self.query'
        # The values are the merged DeltaW matrices for these layers.
        
        final_param_sd = OrderedDict()
        for layer_base_name, merged_matrix_val in matrix_sd.items():
            if layer_base_name.endswith(':other'): # Handle 'other' parameters if any were processed
                original_key = layer_base_name.replace(':other', '')
                if original_key in reference_original_sd:
                    final_param_sd[original_key] = merged_matrix_val.reshape(reference_original_sd[original_key].shape).to(reference_original_sd[original_key].dtype)
                else:
                    print(f"Warning matrix_to_state_dict: Original key for '{layer_base_name}' not found in reference. Skipping.")
                continue

            # For 'weight' type parameters (our DeltaWs)
            if layer_base_name in reference_original_sd:
                # Reshape merged_matrix_val (which might have been flattened or altered by SVDMerger's internal matrix format)
                # back to the original DeltaW matrix shape.
                # The keys from LoRAHandler are already the correct final parameter names for DeltaW.
                final_param_sd[layer_base_name] = merged_matrix_val.reshape(reference_original_sd[layer_base_name].shape).to(reference_original_sd[layer_base_name].dtype)
            else:
                print(f"Warning matrix_to_state_dict: Layer base name '{layer_base_name}' from merged matrices not found in reference SD. Skipping.")

        # Original KnOTS code had logic for splitting bias and handling shared weights,
        # which is not directly applicable if we are only merging LoRA DeltaW matrices
        # that don't have explicit biases or shared embedding tables.

        return final_param_sd
    
    def transform(self, *args, **kwargs):
        # This method is usually implemented by subclasses like SVDMerger
        # to populate self.ingredients.
        raise NotImplementedError("transform() should be implemented by subclasses like SVDMerger.")

class VectorMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models, 
            pretrained_model=pretrained_model, 
            param_handler=param_handler, 
            device=device,
            merge_config=merge_config
        )
        
        self.representation_helper = VectorOps()
    
    def merge(self, merge_config={'merge_method': 'tv'}):
        print(merge_config['merge_method'])
        merging_fn = lambda x: get_merging_fn(merge_config['merge_method'])(
            x, **merge_config, weights=self.scaling_coeffs
        )

        ptm_reference_params = self.param_handler(self.pretrained_model).get_ft_parameters()
        ftms_relevant_params = [ftm.get_ft_parameters() for ftm in self.ftms_params]
        ftms_task_dirs = self.get_task_directions(ptm_reference_params, ftms_relevant_params)
        
        if merge_config.get('dare', False):
            ftms_task_dirs = self.apply_dare(
                ftms_task_dirs, merge_config['dare_pruning_coeffs'], merge_config['dare_seed']
            )
        
        merged_sd = self.representation_helper(ftms_task_dirs, merging_fn, merge_config)

        merged_base = deepcopy(self.pretrained_model)
        if len(merged_sd) == 2:
            merged_sd, mask = merged_sd
        merged_model = self.add_task_parameters(merged_base, merged_sd)
        
        return merged_model

class SVDMerger(TaskMerger):
    def __init__(self, finetuned_models, pretrained_model, param_handler, device=0, merge_config=None):
        super().__init__(
            finetuned_models=finetuned_models, 
            pretrained_model=pretrained_model, 
            param_handler=param_handler, 
            device=device,
            merge_config=merge_config
        )
        
        # After super().__init__():
        # self.ftms_params is now a list of state_dicts [sd1, sd2, ...]
        # self.pt_params is the pretrained state_dict (e.g., zeros_sd)

        if not self.ftms_params: # Check if the list of state_dicts is empty
            raise ValueError("SVDMerger initialization: self.ftms_params (list of state_dicts) is empty. Cannot determine layer names.")
        
        # Ensure the first element is indeed a dictionary (state_dict)
        if not isinstance(self.ftms_params[0], (dict, OrderedDict)):
             raise TypeError(
                 f"SVDMerger expects self.ftms_params to be a list of dicts (state_dicts), "
                 f"but got {type(self.ftms_params[0])} for the first element."
             )

        # self.get_layer_names expects a state_dict and returns a defaultdict:
        # e.g., { 'bert.encoder.layer.0.attention.self.query': {'weight': 'bert.encoder.layer.0.attention.self.query'}, ... }
        # self.layer_names should be a list of unique layer base_names (strings).
        
        # Pass the first state_dict directly to get_layer_names
        layer_names_map = self.get_layer_names(self.ftms_params[0]) 
        
        # Extract the keys from the map (which are the base layer names) and sort them
        self.layer_names = sorted(list(layer_names_map.keys())) 
        
        self.representation_helper = VectorOps()
        self.ingredients = None
            
    def variable_extend_dim(self, elements, op_dim):
        if isinstance(elements, list):
            return [self.variable_extend_dim(element, op_dim) for element in elements]
        while len(elements.shape) < (op_dim+1):
            elements = elements.unsqueeze(-1)
        return elements
    
    def dict_of_concat_matrices(self, list_of_dictmatrices, dim=0, concat_across_output = True):
        dict2matrix_stack = defaultdict(lambda: list())
        for dict2matrix in list_of_dictmatrices:
            for key, val in dict2matrix.items():
                if(concat_across_output == True):
                    dict2matrix_stack[key] += [val.to(self.device)]
                else: 
                    dict2matrix_stack[key] += [val.T.to(self.device)]
                
        for key, list_of_vals in dict2matrix_stack.items():
            # Extend dim as necessary
            list_of_vals = self.variable_extend_dim(list_of_vals, op_dim=dim)
            dict2matrix_stack[key] = torch.concat(list_of_vals, dim=dim)
        return dict2matrix_stack
    
    def reconstruct_merged_sd(self, U_sd, sV_sd):
        if isinstance(sV_sd, list):
            if isinstance(U_sd, list):
                return [self.reconstruct_merged_sd(U, sV) for U, sV in zip(U_sd, sV_sd)]
            return [self.reconstruct_merged_sd(U_sd, sV) for sV in sV_sd]
        sd = {}
        for key, U in U_sd.items():
            sd[key] = (U @ sV_sd[key]).to(torch.float32)
        return sd
        
    def apply_svd(self, ft_params, concat_across_output = True, svd_device='cpu'):
        UsV_dict = {}
        basis_dict = {} # basis for reconstruction
        s_compositions_dict = [dict() for _ in range(len(ft_params))]
        V_compositions_dict = [dict() for _ in range(len(ft_params))] # basis composition information per task
        
        print(f'Calculating SVD over {len(ft_params)} models. SVD on device: {svd_device}. Target SVD precision: float32 (from float64 for stability if needed then cast).')
        concated_ft_params = self.dict_of_concat_matrices(ft_params, dim=1, concat_across_output = concat_across_output)
        for key, val in tqdm(concated_ft_params.items(), desc='Obtaining SVDs...'):
            U, s, V = torch.linalg.svd(val.to(torch.float32), full_matrices=False)
            # Keep only supported basis components
            U = U[:, s > 1e-5].type(torch.float32)
            V = V[s > 1e-5].type(torch.float32)
            s = s[s > 1e-5].type(torch.float32)
            UsV_dict[key] = {'U': deepcopy(U), 's':deepcopy(s), 'V':deepcopy(V) }
            # Set all s to be the same scale
            s[s <= 1e-5] = 0
            cat_hidden_dim = V.shape[1] // len(ft_params)

            basis_dict[key] = U.cpu()
            sV_concat = V
            Vs = list(torch.split(sV_concat, cat_hidden_dim, dim=1))
            for idx, V in enumerate(Vs):
                V = torch.diag(s) @ V # Simple and safe for all merging methods we use.
                s_model = s / s

                s_compositions_dict[idx][key] = s_model.cpu()
                V_compositions_dict[idx][key] = V.cpu()
        return basis_dict, s_compositions_dict, V_compositions_dict, UsV_dict
    
    def apply_Ss_on_Vs(self, task_Vs, task_Ss):
        task_sVs = [dict() for i in range(len(task_Vs))]
        for idx, (Vs, Ss) in enumerate(zip(task_Vs, task_Ss)):
            for key, V in Vs.items():
                if len(Ss[key].shape) == 2:
                    task_sVs[idx][key] = Ss[key] @ V
                else:
                    task_sVs[idx][key] = torch.diag(Ss[key]) @ V
        return task_sVs
    
    def remove_others(self, ftms_mats_list):
        # ftms_mats_list is a list of dicts, e.g., [{'layerA': tensor1, 'layerB': tensor2}, {'layerA': tensor3, 'layerB': tensor4}]
        # Each dict maps layer_base_name to its matrix.
        
        num_tasks = len(ftms_mats_list)
        other_mats_list = [OrderedDict() for _ in range(num_tasks)]
        transform_mats_list = [OrderedDict() for _ in range(num_tasks)] # Use OrderedDict here too
        
        if not ftms_mats_list: # Handle empty input
            print(f'Len other: 0| len: transform: 0 (Input ftms_mats_list was empty)')
            return other_mats_list, transform_mats_list

        for task_idx, task_matrix_dict in enumerate(ftms_mats_list):
            for layer_key, matrix_val in task_matrix_dict.items():
                # layer_key is the base name like 'bert.encoder.layer.0.attention.self.query'
                if ':other' in layer_key: # This condition should be false for our DeltaW keys
                    other_mats_list[task_idx][layer_key] = matrix_val
                # elif 'modules_to_save' in layer_key: # Not applicable for LoRA DeltaW typically
                #     other_mats_list[task_idx][layer_key] = matrix_val
                else: # This is where our DeltaW matrices should go
                    transform_mats_list[task_idx][layer_key] = matrix_val
                    
        # Safe printing
        len_other_first_task = len(other_mats_list[0]) if num_tasks > 0 else 0
        len_transform_first_task = len(transform_mats_list[0]) if num_tasks > 0 else 0
        print(f'Remove_others: Len other[0]: {len_other_first_task}| len transform[0]: {len_transform_first_task}')
        
        return other_mats_list, transform_mats_list
    
    def add_others(self, ftms_mats, ftms_others):
        if isinstance(ftms_mats, list):
            return [self.add_others(ftms_mat, ftms_other) for ftms_mat, ftms_other in zip(ftms_mats, ftms_others)]
        
        for key, val in ftms_others.items():
            ftms_mats[key] = val
        return ftms_mats
    
    def transform(self, merge_config):
        # Setup parameters based on the corrected understanding from SVDMerger.__init__
        # self.pt_params is already the state_dict of pretrained params (e.g., zeros_sd from TaskMerger.__init__)
        # self.ftms_params is already the list of state_dicts of finetuned params (e.g., [delta_w_sd1, ...] from TaskMerger.__init__)
        
        # Make deepcopies to avoid modifying the original state_dicts stored in self.pt_params and self.ftms_params
        ptm_reference_params_for_transform = deepcopy(self.pt_params)
        
        # ftms_relevant_params_for_transform are the actual DeltaW state_dicts, already processed by param_handler in TaskMerger.__init__
        ftms_relevant_params_for_transform = deepcopy(self.ftms_params) 
        
        # Since ftms_relevant_params_for_transform are already DeltaWs 
        # and ptm_reference_params_for_transform is zeros_sd (in your specific use case),
        # get_task_directions will effectively return ftms_relevant_params_for_transform.
        # ftms_task_dirs will be a list of DeltaW state_dicts.
        ftms_task_dirs = self.get_task_directions(
            ptm_reference_params_for_transform, 
            ftms_relevant_params_for_transform
        )
        
        # Convert these DeltaW state_dicts to the matrix format expected by apply_svd
        # ftms_task_mats will be a list of dicts: [{layer_name: matrix}, {layer_name: matrix}, ...]
        ftms_task_mats = self.directions_to_matrices(ftms_task_dirs)
        
        # Separate 'other' parameters if any (though for LoRA DeltaW, this should be empty)
        ftms_others, ftms_mats_for_svd = self.remove_others(ftms_task_mats)

        # Get svd_device from merge_config, default to 'cpu'
        svd_processing_device = merge_config.get('svd_device', 'cpu')
        print(f"DEBUG SVDMerger.transform: Using svd_device='{svd_processing_device}' for apply_svd.")

        # Perform SVD on the matrices
        U, task_Ss, task_sVs, UsV_dict = self.apply_svd(
            ftms_mats_for_svd, # Use the filtered matrices
            concat_across_output = merge_config.get('concat_across_output', True),
            svd_device=svd_processing_device
        )
            
        self.ingredients = {
            # 'ftms_relevant_params': ftms_relevant_params_for_transform, # Storing this might be large and is implicitly used.
                                                                        # It's essentially self.ftms_params.
            'ftms_others': ftms_others, # Store any separated 'other' parameters
            'ptm_reference_params': ptm_reference_params_for_transform, # This is zeros_sd, useful for matrix_to_state_dict later
            'U': U,                     # Basis matrices from SVD
            'task_Ss': task_Ss,         # Scaling components for each task
            'task_sVs': task_sVs,       # sV components for each task (S_i @ V_i)
            'UsV_dict': UsV_dict,       # Full SVD decomposition if needed elsewhere
        }
        
        if merge_config.get('ingredients_path') is not None:
            torch.save(self.ingredients, merge_config['ingredients_path'])
    
    def merge(self, merge_config):
        if merge_config.get('ingredients_path') is not None:
            ingredients = torch.load(merge_config['ingredients_path'])
        else:
            ingredients = deepcopy(self.ingredients)
            
        ftms_others = ingredients['ftms_others']
        ptm_reference_params = ingredients['ptm_reference_params']
        U = ingredients['U']
        task_Ss = ingredients['task_Ss']
        task_sVs = ingredients['task_sVs']
        
        if merge_config.get('dare', False):
            print("Applying DARE")
            task_sVs = self.apply_dare(
                task_sVs, merge_config['dare_pruning_coeffs'], merge_config['dare_seed']
            )
        
        representations = self.representation_helper.directions_to_reps(task_sVs)
        ftms_reps = representations
        
        mask_fn = get_mask_fn(merge_config['merge_method'])
        masks = mask_fn(ftms_reps, **merge_config)
        ftms_reps = torch.vstack(ftms_reps).clone()
        masked_sVs = ftms_reps * masks
        pre_merge_sVs_dict = self.representation_helper.rep_to_state_dict(masked_sVs, task_sVs[0])
        rescaled_Vs = self.apply_Ss_on_Vs(pre_merge_sVs_dict, task_Ss)
        
        rescaled_Vs = torch.stack(self.representation_helper.directions_to_reps(rescaled_Vs), dim=0)
        merged_sV_ = masked_merge(
            merge_func=merge_config.get('merging_type'), vectors=rescaled_Vs, weights=self.scaling_coeffs
        )
        merged_sV_sd = self.representation_helper.rep_to_state_dict(merged_sV_, task_sVs[0])
        
        merged_sd = self.reconstruct_merged_sd(U, merged_sV_sd)
        
        merging_fn = lambda x: get_merging_fn(merge_config['merge_method'])(
            x, **merge_config, weights=self.scaling_coeffs
        )
        if merge_config.get('merge_other_params', False):
            merged_others,_ = self.representation_helper(ftms_others,  merging_fn=merging_fn)
            merged_sd = self.add_others(merged_sd, merged_others)
        
        merged_sd = self.matrix_to_state_dict(merged_sd, ptm_reference_params)
        # Add merged sd to the ptm
        merged_base = deepcopy(self.pretrained_model)
        merged_model = self.add_task_parameters(merged_base, merged_sd,  concat_across_output = merge_config.get('concat_across_output', True))
        return merged_model
    

def get_merge_handler(rep_type):
    if rep_type == 'svd-vector':
        return SVDMerger
    elif rep_type == 'vector':
        return VectorMerger
