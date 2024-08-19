import jiwer

from jiwer.transforms import RemoveKaldiNonWords
from lhotse.recipes.chime6 import TimeFormatConverter, normalize_text_chime6
import json
import os

from preprocessing import get_formated_date


def chime_normalisation(input:str) -> str:
    jiwer_chime6_scoring = jiwer.Compose(
    [
        RemoveKaldiNonWords(),
        jiwer.SubstituteRegexes({r"\"": " ", "^[ \t]+|[ \t]+$": "", r"\u2019": "'"}),
        jiwer.RemoveEmptyStrings(),
        jiwer.RemoveMultipleSpaces(),
    ])
    jiwer_chime7_scoring = jiwer.Compose(
    [
        jiwer.SubstituteRegexes(
            {
                "(?:^|(?<= ))(hm|hmm|mhm|mmh|mmm)(?:(?= )|$)": "hmmm",
                "(?:^|(?<= ))(uhm|um|umm|umh|ummh)(?:(?= )|$)": "ummm",
                "(?:^|(?<= ))(uh|uhh)(?:(?= )|$)": "uhhh",
            }
        ),
        jiwer.RemoveEmptyStrings(),
        jiwer.RemoveMultipleSpaces(),
    ])
    def chime6_norm_scoring(txt):
        return jiwer_chime6_scoring(normalize_text_chime6(txt, normalize="kaldi"))


# here we also normalize non-words sounds such as hmmm which are quite a lot !
# you are free to use whatever normalization you prefer for training but this
# normalization below will be used when we score your submissions.
    def chime7_norm_scoring(txt):
        return jiwer_chime7_scoring(
            jiwer_chime6_scoring(
                normalize_text_chime6(txt, normalize="kaldi")
            )  # noqa: E731
        )  # noqa: E731
    return chime7_norm_scoring(input)

def compute_metrics(pred):
    from Whisper import tokenizer, metric
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)


    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
def compute_chime_metrics(pred):
    from Whisper import model_id, dataset_name, version, tokenizer
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    results = {"predictions": pred_str, "labels": label_str}
    results_directory = str(f"{model_id}_{dataset_name}_{version}_{get_formated_date()}")
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    file_path = os.path.join(results_directory, "results.json")
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

# Define the file path


# Write the evaluation results to the file
    # Example evaluation results
    chime_normalized_reference = [chime_normalisation(reference) for reference in label_str]
    chime_normalized_prediction = [chime_normalisation(pred) for pred in pred_str]

    #wer = 100 * metric.compute(predictions=chime_normalized_prediction, references=chime_normalized_reference)
    wer = jiwer.wer(list(chime_normalized_prediction), list(chime_normalized_reference))

    return {"wer": wer}


import meeteval
from meeteval.viz.visualize import AlignmentVisualization

folder = r'https://raw.githubusercontent.com/fgnt/meeteval/main/'
av = AlignmentVisualization(
    meeteval.io.load(folder + 'example_files/ref.stm').groupby('filename')['recordingA'],
    meeteval.io.load(folder + 'example_files/hyp.stm').groupby('filename')['recordingA']
)
#display(av)  # Jupyter
#av.dump('viz.html')  # Create standalone HTML file


# In[ ]:


import meeteval
import pandas as pd
import jiwer
from jiwer.transforms import RemoveKaldiNonWords
from lhotse.recipes.chime6 import TimeFormatConverter, normalize_text_chime6

# SISO WER
wer = meeteval.wer.wer.siso.siso_word_error_rate(
    reference='The quick brown fox jumps over the lazy dog',
    hypothesis='The qwick brown fox jump over lazy '
)
print(wer)