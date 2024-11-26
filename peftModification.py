from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch
import os
# Define LoRA Config
# Print out all the module names in the model
import bitsandbytes as bnb
from transformers import WhisperForConditionalGeneration


def create_peft_model(model):
    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()  # reduce number of stored activations
    model.enable_input_require_grads()

    for name, module in model.named_modules():
        print(name)
    linear_modules = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    target_modules = [name for name in linear_modules if name.startswith('model.encoder.layers')]
    # lowering alpha seems to be helpful when training although keeping r significantly higher than alpha does not seem to work either.
    text_lora_config = LoraConfig(
        r=2,
        lora_alpha=2,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        bias="none",
        task_type="Seq2Seq",
    )
    # prepare int-8 model for training
    model = prepare_model_for_kbit_training(model)
    print([module for module in model.modules()])
    # add LoRA adaptor
    model = get_peft_model(model, text_lora_config)
    model.print_trainable_parameters()
    return model

def create_peft(run_details):
    from peft import prepare_model_for_kbit_training
    from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
    model = WhisperForConditionalGeneration.from_pretrained( "reach-vb/whisper-large-v2-hindi-100steps", load_in_8bit=True, device_map="auto" )
    model = prepare_model_for_kbit_training(model)

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_( True )

    model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
    # Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
    config = LoraConfig( r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none" )

    model = get_peft_model( model, config )
    model.print_trainable_parameters()
    return model


def alterative_peft(run_details, model):
    from peft import prepare_model_for_kbit_training
    from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
    from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
    # q_proj, v_proj, k_proj, out_proj, fc1, fc2
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", layers_to_transform=[0,1])

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model 
  


# This callback helps to save only the adapter weights and remove the base model weights.



