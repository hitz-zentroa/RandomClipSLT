import torch
from sacrebleu.metrics import BLEU


def decode_sentence(sentence):
    byte_tokens = sentence[(sentence >= 3) & (sentence <= 258)]
    return bytes((byte_tokens - 3).tolist()).decode('utf-8', errors='ignore')


def decode_and_clean(sentence):
    return decode_sentence(sentence).replace('\n', ' ')


def decode_with_tokenizer_and_clean(sentence, tokenizer):
    if isinstance(sentence, torch.Tensor):
        sentence = sentence.tolist()
    return tokenizer.decode(sentence, skip_special_tokens=True).replace('\n', ' ')


def compute_bleu_from_token_ids(predictions, labels, model_name=None, tokenizer=None):
    labels = labels.copy()
    labels[labels == -100] = 0

    use_byt5_decoder = model_name is None or 'byt5' in model_name.lower()

    if use_byt5_decoder:
        decoded_predictions = [decode_and_clean(pred) for pred in predictions]
        decoded_labels = [decode_and_clean(label) for label in labels]
    else:
        if tokenizer is None:
            raise ValueError('Tokenizer is required when model_name does not contain "byt5".')
        predictions = predictions.copy()
        predictions[predictions == -100] = 0
        decoded_predictions = [decode_with_tokenizer_and_clean(pred, tokenizer) for pred in predictions]
        decoded_labels = [decode_with_tokenizer_and_clean(label, tokenizer) for label in labels]

    bleu = BLEU()

    return bleu.corpus_score(decoded_predictions, [decoded_labels]).score
