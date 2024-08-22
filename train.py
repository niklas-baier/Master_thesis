from dataclasses import dataclass
from datetime import datetime
from typing import final, Final
import pandas as pd
from transformers import WhisperTokenizer


@dataclass
@final
class RunDetails:
    dataset_name: str #name of the dataset
    model_id: str #name of the model
    version: str  # plain model or modifed ?
    environment: str # laptop or cluster
    train_state: str # training wanted ?
    date: str # current date
    device: str # cuda
    task: str #classification or transciption or joint
    developer_mode: str # small datasets?

@dataclass
class DataDetails:
    num_speakers: int
    speakers: [str]
    num_origins: int
    origins: [str]
    num_locations: int
    locations: [str]
    ref_chimes: [str]
    num_ref_chimes: int
    ref_dipcos: [str]
    num_ref_dipcos: int
    session_ids: [str]
    num_session_ids: int
    genders : [str]
    num_genders: int
    nativitys: [str]
    num_nativitys: int
    mother_tongues: [str]
    num_mother_tongues: int

def generate_data_details(dataframe):
    arguments_datadetails = {}
    special_columns = [
        'speaker', 'origin', 'location', 'ref_chime', 'ref_dipco',
        'session_id', 'gender', 'nativity', 'mother_tongue'
    ]

    for x in special_columns:
        if x in dataframe.columns:
            versions_x = dataframe[x].unique()
            plural_key = x + "s"
            arguments_datadetails[plural_key] = versions_x

            num_x = len(versions_x)
            num_key = "num_" + plural_key
            arguments_datadetails[num_key] = num_x

    data_details = DataDetails(arguments_datadetails)
    return data_details


def trained_model_transcription(model, eval_dataset, Run_details):
    tokenizer = WhisperTokenizer.from_pretrained(RunDetails.model_id, task="transcribe", language="en")
    from Whisper import processor
    # Load the tokenizer (if necessary)

    # Example input

    from tqdm import tqdm

    # Iterate over the dataset with progress tracking
    eval_temp = pd.DataFrame(columns=['results_trained'])
    for i, example in tqdm(enumerate(eval_dataset), total=len(eval_dataset)):
        sample = eval_dataset[0]["audio"]
        input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"],
                                   return_tensors="pt").input_features
        input_features = input_features.to(Run_details.device)
        outputs = model.generate(input_features)
        transcription = processor.batch_decode(outputs, skip_special_tokens=True)

        eval_temp.loc[i, "results_trained"] = transcription




from transformers import EncoderDecoderCache
from transformers.models.whisper.modeling_whisper import _CONFIG_FOR_DOC
from transformers.models.whisper.modeling_whisper import *
from typing import Optional, Tuple, Union
import types

'''
class WhisperForConditionalGeneration2(WhisperGenerationMixin, WhisperPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model = WhisperModel(config)
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
        #self.num_speakers=4
        #self.classifier = nn.Linear(config.vocab_size,self.num_speakers)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.model.encoder._freeze_parameters()

    @add_start_docstrings_to_model_forward(WHISPER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.proj_out(outputs[0])
        #speaker_logits = self.classifier(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels_clas = labels.to(lm_logits.device)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))
            #loss_classification = lm_logits.view(-1, self.num_speakers)
            #weighted_loss =  0.8 * loss + 0.2 * loss_classification

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        use_cache=None,
        encoder_outputs=None,
        attention_mask=None,
        decoder_attention_mask=None,
        cache_position=None,
        **kwargs,
    ):
        decoder_position_ids = None
        if decoder_attention_mask is not None:
            decoder_position_ids = (decoder_attention_mask.cumsum(-1) - 1).clamp(min=0)

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
            else:
                past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

            if decoder_position_ids is not None and decoder_position_ids.shape[1] > decoder_input_ids.shape[1]:
                decoder_position_ids = decoder_position_ids[:, remove_prefix_length:]

        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + decoder_input_ids.shape[1], device=decoder_input_ids.device
            )
        elif use_cache:
            cache_position = cache_position[-decoder_input_ids.shape[1] :]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "use_cache": use_cache,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_position_ids": decoder_position_ids,
            "cache_position": cache_position,
        }


'''

def generate_training_args(run_details):
    train_batch_size = 16
    per_device_eval_batch_size = 16
    max_steps = 300
    loggings_steps = 100
    save_steps = 200
    if run_details.environment == 'cluster':
        if 'tiny' in run_details.model_id:
            train_batch_size = 64
            per_device_eval_batch_size = 64
            max_steps = 4000
    return train_batch_size, per_device_eval_batch_size, max_steps, save_steps


