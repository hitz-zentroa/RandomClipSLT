import argparse
import os
import json
import torch
import yaml

from data.data import T5_NEWLINE_TOKEN, DataCollator
from data.sentence_datasets import (DATASETS, dataset_path, protocol_name,
                                    selection_key, load_sentence_dataset, sentence_metadata)
from data.keypoint_processing import get_processed_keypoint_dim
from eval_utils import compute_bleu_from_predictions, compute_generation_metrics, get_bleu_references
from model import CustomByT5Model

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, Adafactor, set_seed


def finetune(args, config):

    seed = config['training'].get('seed', 42)
    set_seed(seed)
    print(f'Random seed: {seed}')

    model_name = config['model']['model_name']
    save_path = config['model']['save_path']
    h5_path = dataset_path(config, args.dataset)
    protocol = protocol_name(args.dataset, args.split_protocol)
    keypoint_config = config['data']['keypoints']
    generation_max_length = config['generation']['max_length']
    max_length_config = config['training']['max_lengths']
    target_effective_batch_size = config['training']['finetune_target_effective_batch_size']
    per_device_batch_sizes = config['training']['finetune_per_device_batch_sizes']
    lr = float(config['training']['finetune_lr'])
    finetune_overrides = config['training'].get('finetune_dataset_overrides', {}).get(args.dataset, {})
    max_steps = finetune_overrides.get('max_steps', config['training']['finetune_max_steps'])
    eval_steps = finetune_overrides.get('eval_steps', config['training'].get('finetune_eval_steps', 150))

    train_ds, train_manifest, train_metadata = load_sentence_dataset(
        h5_path, args.dataset, args.train_split, args.tgt_lang, protocol,
    )

    print(f'Training dataset loaded with {len(train_ds)} instances.')

    val_ds, val_manifest, val_metadata = load_sentence_dataset(
        h5_path, args.dataset, args.val_split, args.tgt_lang, protocol,
    )

    print(f'Validation dataset loaded with {len(val_ds)} instances.')

    if set(train_ds['sentence_id']) & set(val_ds['sentence_id']):
        raise ValueError('Training and validation contain overlapping sentences.')
    data_paths_dict = {selection_key(args.dataset, protocol, part): h5_path
                       for part in (args.train_split, args.val_split)}
    selection = {'dataset': args.dataset, 'protocol': protocol,
                 'tgt_lang': train_ds[0]['tgt_lang'], 'hdf5_path': h5_path,
                 'train': train_manifest, 'validation': val_manifest,
                 'train_metadata': train_metadata, 'validation_metadata': val_metadata,
                 'checkpoint_path': args.checkpoint_path, 'config': config}
    run_dir = os.path.join(save_path, args.run_name)
    selection_path = os.path.join(run_dir, 'data_selection.json')
    if args.resume:
        with open(selection_path, encoding='utf-8') as file:
            saved_selection = json.load(file)
        # Older manifests included attributes that no longer belong to the format.
        for part in ('train', 'validation'):
            saved_selection[part] = [
                {'sentence_id': row['sentence_id'], **sentence_metadata(row)}
                for row in saved_selection[part]
            ]
        if saved_selection != selection:
            raise ValueError('Resume configuration/data selection differs from the saved run.')
    elif int(os.environ.get('RANK', 0)) == 0:
        os.makedirs(run_dir, exist_ok=True)
        with open(selection_path, 'x', encoding='utf-8') as file:
            json.dump(selection, file, ensure_ascii=False, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    references = get_bleu_references(val_ds)

    def compute_metrics(eval_pred):
        predictions, _ = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Distributed evaluation pads the final global batch to a multiple of
        # the per-device batch size. In the currently installed
        # Transformers/Accelerate versions those extra examples can reach
        # compute_metrics, so remove them before computing corpus metrics.
        num_validation_samples = len(val_ds)
        predictions = predictions[:num_validation_samples]

        metrics = {
            'bleu': compute_bleu_from_predictions(
                predictions,
                references,
                model_name=model_name,
                tokenizer=tokenizer
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

    input_dim = get_processed_keypoint_dim(keypoint_config)
    print('input_dim:', input_dim)
    
    collator = DataCollator(
        data_paths=data_paths_dict, 
        tokenizer=tokenizer, 
        model_input_dim=input_dim,
        keypoint_config=keypoint_config,
        max_length_config=max_length_config
    )

    model = CustomByT5Model(model_name=model_name, input_dim=input_dim, load_weights=False)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'))

    optimizer = Adafactor(
        model.parameters(),
        lr=lr,
        relative_step=False,
        scale_parameter=False
    )

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if target_effective_batch_size < per_device_batch_sizes['train'] * world_size or target_effective_batch_size % (per_device_batch_sizes['train'] * world_size):
        raise ValueError('Effective batch size must be a positive multiple of the global microbatch size.')
    gradient_accumulation_steps = target_effective_batch_size // (per_device_batch_sizes['train'] * world_size)
    print(f'Gradient accumulation steps: {gradient_accumulation_steps} (target effective batch size: {target_effective_batch_size}, per device batch size: {per_device_batch_sizes["train"]}, world size: {world_size})')

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(save_path, args.run_name),
        run_name=args.run_name,
        seed=seed,
        data_seed=config['training'].get('data_seed', seed),
        logging_steps=min(50, eval_steps),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_sizes['train'],
        per_device_eval_batch_size=per_device_batch_sizes['eval'],
        gradient_accumulation_steps=gradient_accumulation_steps,
        #learning_rate=5e-4,#1e-3,
        optim="adafactor",
        lr_scheduler_type='constant',#'constant_with_warmup',
        # max_grad_norm=0,
        # warmup_steps=1000,
        eval_strategy='steps',
        eval_on_start=True,
        save_strategy='steps',
        eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=2,
        metric_for_best_model='eval_bleu',
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        generation_num_beams=1,
        bf16=True,
        remove_unused_columns=False,
        save_safetensors=False,
        dataloader_num_workers=16
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        optimizers=(optimizer, None),
        compute_metrics=compute_metrics
    )

    trainer.train(resume_from_checkpoint=args.resume)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--config-path', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dataset', default='how2sign', choices=DATASETS)
    parser.add_argument('--split-protocol', help='FLORES-LSE: zs (default), si, sd; LSE-Health: sd_td (default), sd_ti, si_ti.')
    parser.add_argument('--train-split', default='train')
    parser.add_argument('--val-split', default='val')
    parser.add_argument('--tgt-lang')
    args = parser.parse_args()

    print(args)

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    print(config)

    finetune(args, config)
