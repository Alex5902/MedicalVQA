import torch.nn as nn
from collections import defaultdict, OrderedDict

"""
True base_model.model.classifier.original_module.dense.weight
True base_model.model.classifier.original_module.dense.bias
True base_model.model.classifier.original_module.out_proj.weight
True base_model.model.classifier.original_module.out_proj.bias

"""


class LoRAHandler(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
    def get_ft_parameters(self):
            layer2lora_parameters = defaultdict(lambda: dict())
            print("DEBUG LoRAHandler: Keys in PeftModel state_dict (first 10 with 'lora'):")
            count = 0
            for k_debug in self.base_model.state_dict().keys():
                if "lora" in k_debug:
                    print(f"  {k_debug}")
                    count += 1
                    if count >= 10:
                        break
            if count == 0:
                print("  No keys with 'lora' found in PeftModel state_dict.")


            for key, val in self.base_model.state_dict().items():
                if key.endswith('.lora_A.default.weight'):
                    if hasattr(str, 'removesuffix'):
                        base_name = key.removesuffix('.lora_A.default.weight')
                    else: 
                        base_name = key[:-len('.lora_A.default.weight')]
                    layer2lora_parameters[base_name]['A'] = val
                elif key.endswith('.lora_B.default.weight'):
                    if hasattr(str, 'removesuffix'):
                        base_name = key.removesuffix('.lora_B.default.weight')
                    else:
                        base_name = key[:-len('.lora_B.default.weight')]
                    layer2lora_parameters[base_name]['B'] = val
            
            task_parameters = OrderedDict() 
            
            if not layer2lora_parameters:
                print("WARNING LoRAHandler: No LoRA A or B weights found in the model's state_dict. Check LoRA configuration and target_modules.", file=sys.stderr)
                return task_parameters 

            for name in sorted(layer2lora_parameters.keys()):
                key2val = layer2lora_parameters[name]
                if 'A' in key2val and 'B' in key2val:
                    lora_A = key2val['A']
                    lora_B = key2val['B']

                    if lora_A.device != lora_B.device:
                        print(f"Warning LoRAHandler: LoRA A and B for layer '{name}' are on different devices ({lora_A.device} vs {lora_B.device}). Moving A to B's device.", file=sys.stderr)
                        lora_A = lora_A.to(lora_B.device)

                    task_parameters[name] = (lora_B @ lora_A)
                else:
                    print(f"WARNING LoRAHandler: Missing LoRA 'A' or 'B' matrix for layer base_name '{name}'. Found components: {list(key2val.keys())}. This layer will be skipped for DeltaW calculation.", file=sys.stderr)
            
            if not task_parameters:
                print("WARNING LoRAHandler: get_ft_parameters() is returning an empty OrderedDict after processing. No complete LoRA (A and B) pairs were found and processed.", file=sys.stderr)
                
            return task_parameters
    
    def get_model(self):
        return self.base_model.get_base_model
    
    
class FFTHandler(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def get_ft_parameters(self):
        return OrderedDict(sorted(self.base_model.state_dict().items()))
    
    def get_final_model(self, **kwargs):
        return self.base_model


class GeneralHandler(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def get_ft_parameters(self):
        return OrderedDict(sorted(self.base_model.state_dict().items()))
    
    def get_final_model(self, **kwargs):
        return self.base_model