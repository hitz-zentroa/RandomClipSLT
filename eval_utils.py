import torch
from sacrebleu.metrics import BLEU


def replace_negative_token_ids(token_ids, pad_token_id=0):
    token_ids = token_ids.copy()
    token_ids[token_ids < 0] = pad_token_id
    return token_ids


def decode_sentence(sentence):
    byte_tokens = sentence[(sentence >= 3) & (sentence <= 258)]
    return bytes((byte_tokens - 3).tolist()).decode('utf-8', errors='ignore')


def decode_and_clean(sentence):
    return decode_sentence(sentence).replace('\n', ' ')


def decode_with_tokenizer_and_clean(sentence, tokenizer):
    if isinstance(sentence, torch.Tensor):
        sentence = sentence.tolist()
    elif hasattr(sentence, 'tolist'):
        sentence = sentence.tolist()
    return tokenizer.decode(sentence, skip_special_tokens=True).replace('\n', ' ')


def decode_token_ids(token_ids, model_name=None, tokenizer=None):
    use_byt5_decoder = model_name is None or 'byt5' in model_name.lower()

    if use_byt5_decoder:
        return [decode_and_clean(token_id_seq) for token_id_seq in token_ids]

    if tokenizer is None:
        raise ValueError('Tokenizer is required when model_name does not contain "byt5".')

    return [
        decode_with_tokenizer_and_clean(token_id_seq, tokenizer)
        for token_id_seq in token_ids
    ]


def compute_bleu_from_token_ids(predictions, labels, model_name=None, tokenizer=None):
    """Legacy label-based scoring, retained for historical notebook comparisons."""
    pad_token_id = tokenizer.pad_token_id if tokenizer is not None else 0
    if pad_token_id is None:
        pad_token_id = 0

    predictions = replace_negative_token_ids(predictions, pad_token_id)
    labels = replace_negative_token_ids(labels, pad_token_id)

    decoded_predictions = decode_token_ids(predictions, model_name, tokenizer)
    decoded_labels = decode_token_ids(labels, model_name, tokenizer)

    bleu = BLEU()

    return bleu.corpus_score(decoded_predictions, [decoded_labels]).score


def get_bleu_references(dataset):
    """Cache complete references in evaluation order, without a tokenizer round-trip."""
    return [text.replace('\n', ' ') for text in dataset['tgt_sentence']]


def score_bleu(predictions, references, metric=None):
    """Return the full SacreBLEU result, rejecting incomplete prediction sets."""
    if len(predictions) != len(references):
        raise ValueError(
            f'Prediction/reference count mismatch: {len(predictions)} predictions '
            f'for {len(references)} references.'
        )
    if metric is None:
        metric = BLEU()
    return metric.corpus_score(predictions, [references])


def compute_bleu_from_predictions(predictions, references, model_name=None, tokenizer=None):
    """Score generated token IDs against complete reference strings, not labels."""
    pad_token_id = tokenizer.pad_token_id if tokenizer is not None else 0
    if pad_token_id is None:
        pad_token_id = 0
    predictions = replace_negative_token_ids(predictions, pad_token_id)
    decoded_predictions = decode_token_ids(predictions, model_name, tokenizer)
    return score_bleu(decoded_predictions, references).score


def compute_generation_metrics(
    token_ids,
    tokenizer,
    generation_max_length,
    model_name=None,
    t5_newline_token='<extra_id_99>',
):
    """Compute corpus-level diagnostics from generated token IDs."""
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0
    token_ids = replace_negative_token_ids(token_ids, pad_token_id)

    total = len(token_ids)
    if total == 0:
        return {
            'num_samples': 0,
            'eos_only_pct': 0.0,
            'hit_max_length_pct': 0.0,
            'newline_pct': 0.0,
            'mean_char_length': 0.0,
        }

    eos_token_id = tokenizer.eos_token_id
    use_byt5_decoder = model_name is None or 'byt5' in model_name.lower()
    newline_token_id = None
    if not use_byt5_decoder:
        newline_token_id = tokenizer.convert_tokens_to_ids(t5_newline_token)

    eos_only = 0
    hit_max_length = 0
    contains_newline = 0
    character_count = 0

    for output in token_ids:
        output_list = output.tolist() if hasattr(output, 'tolist') else list(output)
        eos_index = output_list.index(eos_token_id) if eos_token_id in output_list else None
        content_end = eos_index if eos_index is not None else len(output_list)
        content_ids = output_list[:content_end]
        if pad_token_id is not None:
            content_ids = [token_id for token_id in content_ids if token_id != pad_token_id]

        eos_only += int(eos_index is not None and not content_ids)
        # Generation stops early only upon EOS; without EOS it used the full budget.
        hit_max_length += int(eos_index is None and len(output_list) >= generation_max_length)

        if use_byt5_decoder:
            decoded = decode_sentence(output)
            has_newline = '\n' in decoded
        else:
            decoded = tokenizer.decode(output_list, skip_special_tokens=True)
            has_newline = newline_token_id in output_list or '\n' in decoded

        contains_newline += int(has_newline)
        character_count += len(decoded)

    return {
        'num_samples': total,
        'eos_only_pct': eos_only / total * 100,
        'hit_max_length_pct': hit_max_length / total * 100,
        'newline_pct': contains_newline / total * 100,
        'mean_char_length': character_count / total,
    }
