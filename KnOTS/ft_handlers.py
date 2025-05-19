import sys 
import torch
from torch import nn
from collections import defaultdict, OrderedDict

class LoRAHandler(nn.Module):
    def __init__(self, base_model): # base_model here is the PeftModel instance
        super().__init__()
        self.peft_model = base_model # Rename for clarity
        
    def get_ft_parameters(self):
        layer2lora_parameters = defaultdict(lambda: dict())

        active_adapter_name = None
        # Try to get the active adapter name robustly
        if hasattr(self.peft_model, 'active_adapters') and self.peft_model.active_adapters:
            active_adapter_name = self.peft_model.active_adapters[0] # PEFT >= 0.7.0
        elif hasattr(self.peft_model, 'active_adapter'): # Older PEFT
            if isinstance(self.peft_model.active_adapter, str):
                active_adapter_name = self.peft_model.active_adapter
            elif isinstance(self.peft_model.active_adapter, (list, set)) and len(self.peft_model.active_adapter) == 1:
                 active_adapter_name = list(self.peft_model.active_adapter)[0]
        
        if active_adapter_name is None:
            # Fallback: iterate through peft_config to find an adapter name if only one exists
            if hasattr(self.peft_model, 'peft_config') and isinstance(self.peft_model.peft_config, dict) and len(self.peft_model.peft_config) == 1:
                active_adapter_name = list(self.peft_model.peft_config.keys())[0]
            else:
                print("ERROR LoRAHandler: Could not determine a single active adapter name from PeftModel. Trying 'default'.", file=sys.stderr)
                active_adapter_name = "default" # Default or first adapter

        print(f"DEBUG LoRAHandler: Using active_adapter_name: '{active_adapter_name}' for DeltaW calculation.")
        
        # PEFT model state_dict keys are like: "base_model.model.{module_path}.lora_A.{adapter_name}.weight"
        # We need to strip "base_model.model." AND the LoRA suffix to get the relative module_path.
        peft_prefix_to_strip = "base_model.model."
        
        suffix_A = f'.lora_A.{active_adapter_name}.weight'
        suffix_B = f'.lora_B.{active_adapter_name}.weight'

        # Iterate over the state_dict of the PeftModel
        model_state_dict = self.peft_model.state_dict()

        for key, val in model_state_dict.items():
            relative_key = key
            if relative_key.startswith(peft_prefix_to_strip):
                relative_key = relative_key[len(peft_prefix_to_strip):] # Now module_path.lora_A...

            if relative_key.endswith(suffix_A):
                # module_path is relative_key without the suffix
                module_path = relative_key[:-len(suffix_A)]
                layer2lora_parameters[module_path]['A'] = val
            elif relative_key.endswith(suffix_B):
                module_path = relative_key[:-len(suffix_B)]
                layer2lora_parameters[module_path]['B'] = val
        
        task_parameters = OrderedDict() # This will store {relative_module_path: DeltaW_matrix}
        if not layer2lora_parameters:
            print(f"WARNING LoRAHandler: No LoRA A or B weights found for adapter '{active_adapter_name}'. Check LoRA config and target_modules.", file=sys.stderr)
            return task_parameters

        # Sort by relative module_path for consistent order
        for name in sorted(layer2lora_parameters.keys()): # name is now like "bert.encoder..."
            key2val = layer2lora_parameters[name]
            if 'A' in key2val and 'B' in key2val:
                lora_A = key2val['A']
                lora_B = key2val['B']
                if lora_A.device != lora_B.device: # Ensure devices match for matmul
                    lora_A = lora_A.to(lora_B.device)
                
                # Calculate DeltaW = B @ A (PEFT scaling is handled by alpha/r, not part of DeltaW here)
                # Or, if your LoRAHandler is expected to return the effective DeltaW (scaled):
                # scaling = self.peft_model.peft_config[active_adapter_name].lora_alpha / self.peft_model.peft_config[active_adapter_name].r
                # task_parameters[name] = scaling * (lora_B @ lora_A)
                task_parameters[name] = (lora_B @ lora_A) # Assuming DeltaW = B@A for now
            else:
                print(f"WARNING LoRAHandler: Missing A or B for layer '{name}' (adapter '{active_adapter_name}'). Found: {list(key2val.keys())}.", file=sys.stderr)
        
        if not task_parameters:
            print(f"WARNING LoRAHandler: get_ft_parameters() returning empty for adapter '{active_adapter_name}' after processing.", file=sys.stderr)
        else:
            print(f"DEBUG LoRAHandler: Extracted DeltaW for {len(task_parameters)} layers. Example key: {list(task_parameters.keys())[0]}")
        return task_parameters
    
    def get_model(self): # Renamed self.base_model to self.peft_model
        if hasattr(self.peft_model, 'get_base_model'):
            return self.peft_model.get_base_model()
        print("Warning LoRAHandler: peft_model does not have 'get_base_model()' method. Returning peft_model itself.", file=sys.stderr)
        return self.peft_model
    
    
class FFTHandler(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def get_ft_parameters(self):
        return OrderedDict(sorted(self.base_model.state_dict().items()))
    
    def get_final_model(self, **kwargs):
        return self.base_model


class GeneralHandler(nn.Module):
    def __init__(self, model_or_state_dict):
        super().__init__()
        # Check if the input is already a state_dict (like OrderedDict or dict)
        if isinstance(model_or_state_dict, dict): 
            self._is_state_dict = True
            # Store the state_dict directly. Ensure it's an OrderedDict for consistency.
            self.parameters_sd = OrderedDict(model_or_state_dict)
            self.model_instance = None # No actual model instance if a state_dict is passed
            # print(f"DEBUG GeneralHandler: Initialized with a state_dict. Keys (first 5): {list(self.parameters_sd.keys())[:5]}")
        elif isinstance(model_or_state_dict, torch.nn.Module): # Check if it's a PyTorch model
            self._is_state_dict = False
            self.model_instance = model_or_state_dict # Store the model instance
            self.parameters_sd = None
            # print(f"DEBUG GeneralHandler: Initialized with a model instance of type: {type(self.model_instance)}")
        else:
            raise TypeError(
                f"GeneralHandler expected a PyTorch model (torch.nn.Module) or a state_dict (dict), "
                f"but received type {type(model_or_state_dict)}"
            )
    
    def get_ft_parameters(self):
        if self._is_state_dict:
            # If initialized with a state_dict, return a sorted OrderedDict copy of it.
            # Sorting ensures canonical order if not already sorted.
            # print("DEBUG GeneralHandler: get_ft_parameters returning stored state_dict.")
            return OrderedDict(sorted(self.parameters_sd.items()))
        else:
            # If initialized with a model, get its state_dict, sort, and return as OrderedDict.
            if hasattr(self.model_instance, 'state_dict') and callable(self.model_instance.state_dict):
                # print("DEBUG GeneralHandler: get_ft_parameters calling model_instance.state_dict().")
                return OrderedDict(sorted(self.model_instance.state_dict().items()))
            else:
                # This case should ideally be caught by the __init__ type check
                raise AttributeError(
                    "GeneralHandler was initialized with an object that is not a state_dict "
                    "and has no callable 'state_dict' method."
                )
    
    def get_final_model(self, **kwargs):
        # This method might need adjustment if GeneralHandler is used in contexts
        # where a full model is expected back after being initialized with just a state_dict.
        # For your current SVDMerger usage (where it's mainly for getting parameters),
        # this might not be critical, but good to be aware of.
        if self._is_state_dict:
            print(
                "Warning GeneralHandler.get_final_model: "
                "Handler was initialized with a state_dict, not a full model. Returning None."
            )
            return None # Or raise an error, or return the state_dict if that's ever expected
        return self.model_instance