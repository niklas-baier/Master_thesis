from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModelForSeq2SeqLM, PeftType
import torch
import os
# Define LoRA Config
# Print out all the module names in the model
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

    '''from peft import prepare_model_for_kbit_training
    from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
    from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
    # q_proj, v_proj, k_proj, out_proj, fc1, fc2

    #model = get_peft_model(model, config)
    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj"] )#target_modules=["q_proj", "k_proj"]

    model.add_adapter(lora_config, adapter_name="adapter_1")
    num_of_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)


    #run_details.num_of_trainable_parameters = num_of_trainable_parameters'''
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(model, lora_config)

    # Check trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")


    return model

class PeftModelForWhisperASR(PeftModelForSeq2SeqLM):
    """
    A custom PEFT model for Whisper ASR tasks.
    It overrides the forward method to correctly pass `input_features` to
    the underlying WhisperForConditionalGeneration model,
    as `PeftModelForSeq2SeqLM` defaults to `input_ids`.
    """
    def __init__(self, model, peft_config, adapter_name="default"):
        super().__init__(model, peft_config, adapter_name)
        # We need to ensure the base model is actually WhisperForConditionalGeneration
        # or at least a model that accepts 'input_features'.
        if not isinstance(self.base_model.model, WhisperForConditionalGeneration):
            raise TypeError(
                "Base model for PeftModelForWhisperASR must be an instance of WhisperForConditionalGeneration."
            )

    def forward(
        self,
        # Key change: explicitly accept input_features
        input_features: torch.Tensor = None,
        # Remove input_ids as it's not used by Whisper encoder directly
        # input_ids=None,
        attention_mask: torch.Tensor = None, # For input_features attention mask (can be None for Whisper features)
        inputs_embeds: torch.Tensor = None, # Keep for generality if prompt learning or other
        decoder_input_ids: torch.Tensor = None,
        decoder_attention_mask: torch.Tensor = None,
        decoder_inputs_embeds: torch.Tensor = None,
        labels: torch.Tensor = None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        return_dict: bool = None,
        task_ids: torch.Tensor = None,
        **kwargs,
    ):
        peft_config = self.active_peft_config

        # Handling for non-prompt learning (e.g., LoRA)
        if not peft_config.is_prompt_learning:
            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}

                # --- THE CRUCIAL MODIFICATION HERE ---
                # Instead of input_ids=input_ids, pass input_features=input_features
                return self.base_model(
                    input_features=input_features, # Pass input_features for Whisper encoder
                    attention_mask=attention_mask, # Pass attention_mask if available (from feature extractor)
                    # inputs_embeds=inputs_embeds, # Only if you intend to use it, else remove
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        # The rest of the forward method (for prompt learning types)
        # You'll need to adapt these sections if you plan to use PROMPT_TUNING, P_TUNING, or PREFIX_TUNING
        # with input_features. For LoRA (TaskType.SEQ_2_SEQ_LM, target_modules=["q_proj", "v_proj"]),
        # the above `if not peft_config.is_prompt_learning:` block is the primary path.

        # If you're only using LoRA, you might simplify or just keep the original logic
        # for other PEFT types as they won't be hit. However, if you use other PEFT types
        # that modify input_embeds, you'd need to consider how `input_features`
        # translates to `inputs_embeds` for those paths.
        # For a pure ASR (audio->text) LoRA setup, the above `if` block is enough.

        # For completeness, copying the rest of the original forward and noting where input_ids/inputs_embeds
        # would need careful consideration for ASR input features.
        batch_size = _get_batch_size(input_features, inputs_embeds) # Adapt _get_batch_size if needed
                                                                   # or assume batch_size can be derived from labels/decoder_input_ids

        # If you were doing prompt learning on the encoder side, you'd need to convert
        # input_features to inputs_embeds for this path.
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            kwargs["past_key_values"] = self.get_prompt(batch_size)
            return self.base_model(
                # These lines would need careful re-evaluation for input_features
                input_ids=None, # Ensure input_ids is NOT passed
                input_features=input_features, # Pass input_features
                decoder_input_ids=decoder_input_ids,
                decoder_inputs_embeds=decoder_inputs_embeds,
                **kwargs,
            )
        elif peft_config.peft_type in [PeftType.PROMPT_TUNING, PeftType.P_TUNING]:
            # This path expects inputs_embeds for the encoder side
            # You would need to convert input_features to inputs_embeds here.
            # Example: inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
            # For ASR, this is more complex as you don't have input_ids for encoder.
            raise NotImplementedError(
                "Prompt Tuning/P-Tuning for Whisper's encoder via input_features is not directly supported by this custom wrapper."
                " You might need to implement feature_extractor to embedding conversion."
            )
        else: # This 'else' block also relates to non-LoRA/prompt-learning cases in original PEFT
            # This path often expects input_ids for the encoder or requires inputs_embeds.
            # We must ensure input_features is used for the encoder.
            if inputs_embeds is None:
                # This part is problematic for Whisper as it relies on input_ids for word_embeddings
                # For Whisper, input_features is the direct input to the encoder.
                # You might need to add a conversion layer if you're using this path.
                # For LoRA, this entire 'else' path is usually not taken.
                pass # Or raise error if you expect this path not to be hit for LoRA

            # Ensure all arguments are passed correctly, prioritizing input_features
            # The structure below is from the original, adapt to pass input_features to self.base_model
            return self.base_model(
                input_features=input_features, # Ensure this is passed
                labels=labels,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_inputs_embeds=decoder_inputs_embeds,
                # And other kwargs that the base model might need
                **kwargs
            )

# This callback helps to save only the adapter weights and remove the base model weights.
