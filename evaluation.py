import jiwer

from jiwer.transforms import RemoveKaldiNonWords
from lhotse.recipes.chime6 import normalize_text_chime6
import json
import os
import numpy as np
import evaluate

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
    from whisper_main import tokenizer, metric
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)


    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def compute_classification_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=pred.label_ids)





def analysis_special_tokens(results):
    from preprocessing import extract_special_token
    results['token'] = results.apply(lambda row: extract_special_token(row['words']), axis=1)
    grouped_train = results.groupby(['token'])
    allowed_tokens = ["[noise]", "[laugh]", "[unintelligible]", "No token", "[laughs]"]  #
    mask = results['token'].isin(allowed_tokens)
    # Find the rows where the condition is False
    failed_rows = results[~mask]
    rows_with_tokens = results[mask]
    return grouped_train



