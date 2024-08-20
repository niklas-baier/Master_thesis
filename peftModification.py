from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch
# Define LoRA Config
# Print out all the module names in the model
import bitsandbytes as bnb
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


