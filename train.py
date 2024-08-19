from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from transformers import WhisperTokenizer


@dataclass
class RunDetails:
    dataset_name: str
    model_id: str
    version: str
    environment: str
    train_state: str
    date: str
    device: str


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









