import argparse
import json
import os
import tempfile

import torch
import yaml

from data.data import T5_NEWLINE_TOKEN, DataCollator
from data.keypoint_processing import get_processed_keypoint_dim
from data.sentence_datasets import (DATASETS, dataset_options, load_sentence_dataset,
                                    dataset_path, protocol_name, selection_key)
from sacrebleu.metrics import BLEU
from eval_utils import (
    compute_bleu_from_predictions,
    compute_generation_metrics,
    decode_token_ids,
    get_bleu_references,
    replace_negative_token_ids,
    score_bleu,
)
from model import CustomByT5Model
from transformers import AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments


DEFAULT_EVAL_BATCH_SIZE = 64
MANIFEST_FIELDS = (
    'sentence_id', 'video_id', 'file_name', 'flores_id',
    'source_row', 'start_time', 'end_time',
)


def prediction_array(prediction_output, tokenizer, num_samples=None):
    predictions = prediction_output.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    predictions = replace_negative_token_ids(predictions, pad_token_id)

    if num_samples is not None:
        predictions = predictions[:num_samples]

    return predictions


def main(args, config):
    model_name = config['model']['model_name']
    keypoint_config = config['data']['keypoints']
    generation_max_length = config['generation']['max_length']
    generation_num_beams = args.beam_size
    if generation_num_beams is None:
        generation_num_beams = config['generation']['beam_size']
    max_length_config = config['training']['max_lengths']
    src_lang, tgt_lang = dataset_options(args.dataset, args.split, args.tgt_lang)
    protocol = protocol_name(args.dataset, args.split_protocol)
    dataset_name = selection_key(args.dataset, protocol, args.split)
    h5_path = dataset_path(config, args.dataset, args.h5_path)

    os.makedirs(args.output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_dim = get_processed_keypoint_dim(keypoint_config)
    print('input_dim:', input_dim)

    dataset, manifest, extraction_metadata = load_sentence_dataset(
        h5_path, args.dataset, args.split, tgt_lang, protocol,
    )
    data_paths = {dataset_name: h5_path}
    collator = DataCollator(
        data_paths=data_paths,
        tokenizer=tokenizer,
        model_input_dim=input_dim,
        keypoint_config=keypoint_config,
        max_length_config=max_length_config,
        random_clip_config=config['data'].get('random_clips'),
    )

    model = CustomByT5Model(
        model_name=model_name,
        input_dim=input_dim,
        load_weights=False,
    )
    model.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))

    eval_batch_size = args.eval_batch_size
    if eval_batch_size is None:
        eval_batch_size = DEFAULT_EVAL_BATCH_SIZE

    references = get_bleu_references(dataset)

    def compute_metrics(eval_pred):
        predictions, _ = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        num_evaluation_samples = len(dataset)
        predictions = predictions[:num_evaluation_samples]

        metrics = {
            'bleu': compute_bleu_from_predictions(
                predictions,
                references,
                model_name=model_name,
                tokenizer=tokenizer,
            )
        }
        metrics.update(compute_generation_metrics(
            predictions,
            tokenizer=tokenizer,
            generation_max_length=generation_max_length,
            model_name=model_name,
            t5_newline_token=T5_NEWLINE_TOKEN,
        ))
        return metrics

    training_args = Seq2SeqTrainingArguments(
        output_dir=tempfile.mkdtemp(prefix='eval_predict_', dir='/tmp'),
        per_device_eval_batch_size=eval_batch_size,
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        generation_num_beams=generation_num_beams,
        bf16=False,#torch.cuda.is_available() and not args.cpu,
        use_cpu=args.cpu,
        remove_unused_columns=False,
        report_to=[],
    )

    print('Evaluation path: Seq2SeqTrainer.predict')
    print('split:', args.split)
    print('dataset:', args.dataset)
    print('h5_path:', h5_path)
    print('languages:', src_lang, '->', tgt_lang)
    print('per_device_eval_batch_size:', eval_batch_size)
    print('generation_max_length:', generation_max_length)
    print('generation_num_beams:', training_args.generation_num_beams)
    print('bf16:', training_args.bf16)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    prediction_output = trainer.predict(dataset)
    if not trainer.is_world_process_zero():
        return

    predictions = prediction_array(
        prediction_output,
        tokenizer,
        num_samples=len(dataset),
    )
    out = decode_token_ids(predictions, model_name=model_name, tokenizer=tokenizer)
    ref = references

    metric = BLEU()
    bleu_score = score_bleu(out, ref, metric=metric)
    if len(manifest) != len(out):
        raise ValueError(
            f'Manifest/prediction count mismatch: {len(manifest)} records '
            f'for {len(out)} predictions.'
        )

    print(bleu_score)
    print(prediction_output.metrics)

    with open(os.path.join(args.output_path, 'ref.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(ref) + '\n')

    with open(os.path.join(args.output_path, 'out.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')

    with open(os.path.join(args.output_path, 'manifest.jsonl'), 'w', encoding='utf-8') as f:
        for row, reference, prediction in zip(manifest, ref, out):
            sentence_metadata = {key: row[key] for key in MANIFEST_FIELDS if key in row}
            f.write(json.dumps({**sentence_metadata, 'reference': reference, 'prediction': prediction},
                               ensure_ascii=False) + '\n')
    with open(os.path.join(args.output_path, 'evaluation.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'dataset': args.dataset, 'protocol': protocol, 'split': args.split, 'h5_path': h5_path,
            'src_lang': src_lang, 'tgt_lang': tgt_lang,
            'checkpoint_path': args.checkpoint_path, 'config': config,
            'generation_max_length': generation_max_length,
            'generation_num_beams': generation_num_beams,
            'per_device_eval_batch_size': eval_batch_size,
            'bleu': {
                'score': bleu_score.score,
                'signature': str(metric.get_signature()),
                'ngram_precisions': bleu_score.precisions,
                'ngram_matches': bleu_score.counts,
                'ngram_totals': bleu_score.totals,
                'brevity_penalty': bleu_score.bp,
                'length_ratio': bleu_score.ratio,
                'prediction_length': bleu_score.sys_len,
                'reference_length': bleu_score.ref_len,
            },
            'extraction_metadata': extraction_metadata,
            'metrics': prediction_output.metrics,
        }, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='configs/config.yaml')
    parser.add_argument('--output-path', type=str, default='outputs/default/')
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--dataset', default='how2sign', choices=DATASETS)
    parser.add_argument('--split-protocol', help='FLORES-LSE: zs (default), si, sd; LSE-Health: sd_td (default), sd_ti, si_ti.')
    parser.add_argument('--h5-path', help='Override the configured dataset file.')
    parser.add_argument('--tgt-lang', help='Default: en for How2Sign/FLORES-LSE; es for LSE-Health.')
    parser.add_argument('--split', type=str, required=True,
                        help='Partition in the selected protocol (e.g. val, test).')
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=None,
        help=f'Override per-device eval batch size. Default: {DEFAULT_EVAL_BATCH_SIZE}.',
    )
    parser.add_argument(
        '--beam-size',
        type=int,
        default=None,
        help='Override generation.beam_size from the config. Use 1 for greedy decoding.',
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Run Trainer prediction on CPU.',
    )
    args = parser.parse_args()
    if args.split == 'train':
        parser.error('Evaluation requires a validation or test split.')

    print(args)

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    print(config)

    main(args, config)
