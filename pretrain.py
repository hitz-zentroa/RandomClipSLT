import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # TODO: kendu hemendik

import argparse
import dataclasses
import glob
import shutil
import torch
import yaml

from data.data import read_to_datasets, load_mt_datasets, sentence_level_hdf5_to_dataset, DataCollator
from data.keypoint_processing import get_processed_keypoint_dim
from eval_utils import compute_bleu_from_token_ids
from model import CustomByT5Model

from datasets import concatenate_datasets
from peft import LoraConfig
from transformers import Adafactor, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.trainer_utils import get_last_checkpoint


def data_multipliers(proportions, len_mt, len_caption_clip, len_random_clip):
    
    if proportions['mt'] == 0:
        
        mul_caption = 1

        aim_random = proportions['random_clip'] * len_caption_clip / proportions['caption_clip']
        mul_random = round(aim_random / len_random_clip)

    else:
        # Asumitzen dut MT izango dela beti handiena
        
        aim_caption = proportions['caption_clip'] * len_mt * 2 / proportions['mt']
        mul_caption = round(aim_caption / len_caption_clip)

        aim_random = proportions['random_clip'] * len_mt * 2 / proportions['mt']
        mul_random = round(aim_random / len_random_clip)

    return mul_caption, mul_random


def main(args, config):

    os.environ["WANDB__SERVICE_WAIT"] = "300" # ???

    hdf5_path = config['data']['hdf5_path']
    val_hdf5_path = config['data']['val_hdf5_path'] # TODO: hau aldatzeko dago
    caption_config = config['data']['captions']
    keypoint_config = config['data']['keypoints']
    random_clip_config = config['data']['random_clips']
    model_name = config['model']['model_name']
    lora = config['model']['lora']
    save_path = config['model']['save_path']
    generation_max_length = config['generation']['max_length']
    max_tokens = config['training']['max_tokens']
    initial_data_proportions = config['training']['initial_data_proportions']
    proportions = config['training']['data_proportions']
    target_effective_batch_size = config['training']['target_effective_batch_size']
    initial_per_device_batch_sizes = config['training']['initial_per_device_batch_sizes']
    per_device_batch_sizes = config['training']['per_device_batch_sizes']
    initial_max_steps = config['training']['initial_max_steps']
    max_steps = config['training']['max_steps']

    data_paths = glob.glob(f"{hdf5_path}/*/*.h5")
    sign_langs = [path.split('/')[-2] for path in data_paths]
    data_paths_dict = {lang: path for lang, path in zip(sign_langs, data_paths)}
    print(f"Found {len(data_paths)} HDF5 data files for sign languages: {', '.join(sign_langs)}")
    print('Creating SLT datasets...')

    # TODO: dev hemendik kendu
    (
        train_caption_real,
        dev_caption_real,
        train_caption_synthetic,
        dev_caption_synthetic,
        train_random_real,
        dev_random_real,
        train_random_synthetic,
        dev_random_synthetic,
    ) = read_to_datasets(
        data_paths=data_paths,
        caption_config=caption_config,
        random_clip_config=random_clip_config
        # frame_freq=frame_freq,
        # chunk_seconds=chunk_seconds,
        # max_characters=max_characters,
        # max_seconds=max_seconds,
        # min_seconds=min_seconds,
        # captions_langs=captions_langs
    )

    print()
    print(f'train_caption_real: {len(train_caption_real)} instances')
    print(f'dev_caption_real: {len(dev_caption_real)} instances')
    print(f'train_random_real: {len(train_random_real)} instances')
    print(f'dev_random_real: {len(dev_random_real)} instances')
    print()

    # TODO: config-etik hartu zein izango den dataseta eta hortik hartu ezaugarriak
    val_dataset = sentence_level_hdf5_to_dataset(
        val_hdf5_path, 
        dataset_name='how2sign_val', 
        src_lang='ase', 
        tgt_lang='en'
    )
    data_paths_dict['how2sign_val'] = val_hdf5_path
    print(f'Validation dataset loaded with {len(val_dataset)} instances.')

    print('Creating MT datasets...')

    mt_ds, reverse_mt_ds = load_mt_datasets(
        config["data"]["mt_pairs"],
        max_characters=caption_config["max_characters"]
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        return {
            'bleu': compute_bleu_from_token_ids(
                predictions,
                labels,
                model_name=model_name,
                tokenizer=tokenizer
            )
        }
    
    input_dim = get_processed_keypoint_dim(keypoint_config)
    print('input_dim:', input_dim)

    collator = DataCollator(
        data_paths=data_paths_dict, 
        tokenizer=tokenizer, 
        model_input_dim=input_dim,
        keypoint_config=keypoint_config,
        max_tokens=max_tokens,
        random_clip_config=random_clip_config
    )

    print('Preparing training data...')

    initial_mul_caption, _ = data_multipliers(initial_data_proportions, len(mt_ds), len(train_caption_real), len(train_random_real))
    mul_caption, mul_random = data_multipliers(proportions, len(mt_ds), len(train_caption_real), len(train_random_real))

    print()
    print('Multipliers:')
    print(f'initial_mul_caption: {initial_mul_caption}')
    print(f'mul_caption: {mul_caption}')
    print(f'mul_random: {mul_random}')
    print()

    initial_oversampled_train_caption_real = concatenate_datasets([train_caption_real] * initial_mul_caption)

    if initial_data_proportions['mt'] == 0:
        initial_train = initial_oversampled_train_caption_real
    else:
        initial_train = concatenate_datasets([initial_oversampled_train_caption_real, mt_ds, reverse_mt_ds])

    oversampled_train_caption_real = concatenate_datasets([train_caption_real] * mul_caption)
    oversampled_train_random_real = concatenate_datasets([train_random_real] * mul_random)

    if proportions['mt'] == 0:
        train = concatenate_datasets([oversampled_train_caption_real, oversampled_train_random_real])
    else:
        train = concatenate_datasets([oversampled_train_caption_real, oversampled_train_random_real, mt_ds, reverse_mt_ds])

    # DEBUG
    '''
    print('---- MT ----')
    print()
    batch_mt = collator([mt_ds[i] for i in range(5)])
    batch_reverse_mt = collator([reverse_mt_ds[i] for i in range(5)])
    print('---- Caption clip ----')
    print()
    batch = collator([train_caption_real[i] for i in range(5)])
    print('---- Random clip ----')
    print()
    batch2 = collator([train_random_real[i] for i in range(20)])

    print('len(train_caption_real):', len(train_caption_real))
    print('len(train_random_real):', len(train_random_real))
    '''

    # KOMENTARIO BEZALA JARRI HEMENDIK AURRERAKO GUZTIA BATCHA BEGIRATU ETA UZTEKO

    if lora:
        lora_config = LoraConfig(
            target_modules=['q', 'k', 'v', 'o'],
            modules_to_save=['lm_head']
        )
    else:
        lora_config = None

    load_weights = not args.resume and not args.only_second

    model = CustomByT5Model(model_name=model_name, input_dim=input_dim, lora_config=lora_config, load_weights=load_weights)

    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        relative_step=False,
        scale_parameter=False
    )

    world_size = int(os.environ.get('WORLD_SIZE', 1))

    gradient_accumulation_steps = target_effective_batch_size // (per_device_batch_sizes['train'] * world_size)
    print(f'Gradient accumulation steps: {gradient_accumulation_steps} (target effective batch size: {target_effective_batch_size}, per device batch size: {per_device_batch_sizes["train"]}, world size: {world_size})')

    training_args = Seq2SeqTrainingArguments(
        os.path.join(save_path, args.run_name),
        logging_steps=100,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_sizes['train'],
        per_device_eval_batch_size=per_device_batch_sizes['eval'],
        gradient_accumulation_steps=gradient_accumulation_steps,
        #learning_rate=1e-3,
        #optim='adafactor',
        lr_scheduler_type='constant_with_warmup',#"inverse_sqrt",#'constant',
        warmup_steps=1000,
        #max_grad_norm=0,
        predict_with_generate=True,
        generation_max_length=generation_max_length,
        eval_strategy='steps',
        save_strategy='steps',
        eval_steps=2000,
        save_steps=2000,
        save_total_limit=2, # bestela bukaeran azkena borratzen du
        metric_for_best_model='eval_bleu',
        #report_to='wandb', # jarri gabe ere egiten du
        run_name=args.run_name,
        #prediction_loss_only=True,
        #fp16=True,
        bf16=True,
        remove_unused_columns=False, # https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76947480#76947480
        save_safetensors=False, # https://discuss.huggingface.co/t/saving-model-in-safetensors-format-through-trainer-fails-for-gemma-2-due-to-shared-tensors/109451
        dataloader_num_workers=18#16
    ) # beste batzuk zeuden hemen: https://medium.com/@anyuanay/fine-tuning-the-pre-trained-t5-small-model-in-hugging-face-for-text-summarization-3d48eb3c4360

    if not args.only_second:

        initial_gradient_accumulation_steps = target_effective_batch_size // (initial_per_device_batch_sizes['train'] * world_size)
        print(f'Initial gradient accumulation steps: {initial_gradient_accumulation_steps} (target effective batch size: {target_effective_batch_size}, initial per device batch size: {initial_per_device_batch_sizes["train"]}, world size: {world_size})')

        # TODO: run_name=args.run_name+'_initial'? Bestela wandb-n grafikoak arraro agertuko dira dena segidan exekutatzean
        initial_training_args = dataclasses.replace(
            training_args,
            output_dir=os.path.join(save_path, args.run_name+'_initial'),
            per_device_train_batch_size=initial_per_device_batch_sizes['train'],
            per_device_eval_batch_size=initial_per_device_batch_sizes['eval'],
            gradient_accumulation_steps=initial_gradient_accumulation_steps,
            max_steps=initial_max_steps,
        )

        initial_trainer = Seq2SeqTrainer(
            model=model,
            args=initial_training_args,
            train_dataset=initial_train,
            eval_dataset=val_dataset, #dev_caption_real,
            data_collator=collator,
            optimizers=(optimizer, None),
            compute_metrics=compute_metrics
    )

        initial_trainer.train(resume_from_checkpoint=args.resume)

        shutil.copytree(
            src=os.path.join(save_path, args.run_name),
            dst=os.path.join(save_path, f'{args.run_name}_initial')
        )

    if args.only_second and not args.resume:
        last_checkpoint = get_last_checkpoint(os.path.join(save_path, args.run_name+'_initial'))
        model.load_state_dict(torch.load(os.path.join(last_checkpoint, 'pytorch_model.bin')))

    if not args.only_initial:

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=val_dataset, #{'caption': dev_caption_real, 'random': dev_random_real},
            data_collator=collator,
            optimizers=(optimizer, None),
            compute_metrics=compute_metrics
        )

        trainer.train(resume_from_checkpoint=args.resume)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a CustomByT5Model.')
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--config-path', type=str, default='configs/config.yaml')
    parser.add_argument('--only-initial', action='store_true')
    parser.add_argument('--only-second', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    print(args)

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    
    main(args, config)
