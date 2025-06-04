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


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import warnings
from train import DataCollatorSpeechSeq2SeqWithPadding, create_tokenizer_model_processor

def calculate_wer_on_dataset(model, dataset, processor, device,run_details, batch_size=8, max_batches=None,
                           use_whisper_normalizer=True, verbose=True, torch_dtype=torch.float32):
    """
    Calculate WER on a dataset using Whisper's text normalizer.

    Args:
        model: Whisper model
        dataset: Dataset to evaluate on
        processor: Whisper processor
        device: Device to run on
        batch_size: Batch size for evaluation
        max_batches: Maximum number of batches to evaluate (None for all)
        use_whisper_normalizer: Whether to use Whisper's BasicTextNormalizer
        verbose: Whether to print detailed progress

    Returns:
        dict: Contains WER, individual predictions, and references
    """

    # Initialize WER metric and normalizer
    wer_metric = evaluate.load("wer")
    normalizer = BasicTextNormalizer() if use_whisper_normalizer else None
    model_name = model.config._name_or_path
    # Create dataloader
    _,model, processor = create_tokenizer_model_processor(run_details, torch_dtype=torch_dtype)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor,model.config.decoder_start_token_id )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for debugging
        collate_fn=collator
    )

    model.eval()
    all_predictions = []
    all_references = []
    total_batches = len(dataloader)

    if max_batches:
        total_batches = min(total_batches, max_batches)

    if verbose:
        print(f"Evaluating WER on {total_batches} batches (batch_size={batch_size})")
        print(f"Using Whisper normalizer: {use_whisper_normalizer}")

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=total_batches, desc="Calculating WER")

        for batch_idx, batch in pbar:
            if max_batches and batch_idx >= max_batches:
                break

            # Move to device
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)

            # Generate predictions
            try:
                # Use generate for better results than teacher forcing
                predicted_ids = model.generate(
                    input_features,
                    max_length=448,  # Whisper's max length
                    num_beams=1,     # Greedy decoding for speed
                    do_sample=False,
                    task="transcribe",
                    language="english" if hasattr(processor.tokenizer, 'language') else None
                )

            except Exception as e:
                if verbose:
                    print(f"Generation failed for batch {batch_idx}: {e}")
                # Fallback to teacher forcing
                outputs = model(input_features=input_features, labels=labels)
                predicted_ids = torch.argmax(outputs.logits, dim=-1)

            # Decode predictions
            # Skip special tokens and decode
            predicted_ids = predicted_ids.cpu()
            predictions = processor.batch_decode(predicted_ids, skip_special_tokens=True)

            # Decode references (labels)
            # Replace -100 with pad token for decoding
            labels_for_decoding = labels.cpu()
            labels_for_decoding = torch.where(
                labels_for_decoding != -100,
                labels_for_decoding,
                processor.tokenizer.pad_token_id
            )
            references = processor.batch_decode(labels_for_decoding, skip_special_tokens=True)

            # Apply Whisper normalizer if requested
            if normalizer:
                normalized_predictions = [normalizer(pred) for pred in predictions]
                normalized_references = [normalizer(ref) for ref in references]
            else:
                normalized_predictions = predictions
                normalized_references = references

            # Store results
            all_predictions.extend(normalized_predictions)
            all_references.extend(normalized_references)

            # Show progress with sample
            if verbose and batch_idx < 3:  # Show first few examples
                print(f"\nBatch {batch_idx} samples:")
                for i in range(min(2, len(predictions))):  # Show 2 samples per batch
                    print(f"  Reference: '{references[i]}'")
                    print(f"  Prediction: '{predictions[i]}'")
                    if normalizer:
                        print(f"  Norm Ref: '{normalized_references[i]}'")
                        print(f"  Norm Pred: '{normalized_predictions[i]}'")
                    print()

    # Calculate WER
    try:
        wer_score = wer_metric.compute(predictions=all_predictions, references=all_references)
        wer_percentage = wer_score * 100
    except Exception as e:
        print(f"Error calculating WER: {e}")
        wer_score = float('inf')
        wer_percentage = float('inf')

    # Calculate additional metrics
    total_samples = len(all_predictions)
    empty_predictions = sum(1 for pred in all_predictions if len(pred.strip()) == 0)
    empty_references = sum(1 for ref in all_references if len(ref.strip()) == 0)

    results = {
        'wer': wer_score,
        'wer_percentage': wer_percentage,
        'total_samples': total_samples,
        'empty_predictions': empty_predictions,
        'empty_references': empty_references,
        'predictions': all_predictions,
        'references': all_references
    }

    if verbose:
        print(f"\n=== WER Evaluation Results ===")
        print(f"Total samples: {total_samples}")
        print(f"WER: {wer_percentage:.2f}%")
        print(f"Empty predictions: {empty_predictions}/{total_samples}")
        print(f"Empty references: {empty_references}/{total_samples}")

        # Show some examples of errors
        print(f"\n=== Sample Comparisons ===")
        for i in range(min(5, len(all_predictions))):
            print(f"Sample {i+1}:")
            print(f"  REF: '{all_references[i]}'")
            print(f"  HYP: '{all_predictions[i]}'")
            # Simple word-level comparison
            ref_words = all_references[i].split()
            hyp_words = all_predictions[i].split()
            if ref_words != hyp_words:
                print(f"  DIFF: {len(ref_words)} ref words vs {len(hyp_words)} hyp words")
            print()

    return results
