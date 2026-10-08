import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" # TODO: kendu hemendik

import argparse
import dataclasses
import glob
import torch
import yaml

from data.data import T5_NEWLINE_TOKEN, read_to_datasets, load_mt_datasets, sentence_level_hdf5_to_dataset, DataCollator
from data.keypoint_processing import get_processed_keypoint_dim
from eval_utils import compute_bleu_from_predictions, compute_generation_metrics, get_bleu_references
from gradient_diagnostics import (
    GradientDiagnosticCollator,
    GradientDiagnosticSeq2SeqTrainer,
)
from model import CustomByT5Model

from datasets import concatenate_datasets
from peft import LoraConfig
from transformers import Adafactor, AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, set_seed
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

    seed = config['training'].get('seed', 42)
    data_seed = config['training'].get('data_seed', seed)
    set_seed(seed)
    print(f'Random seed: {seed}')
    print(f'Data seed: {data_seed}')

    hdf5_path = config['data']['hdf5_path']
    val_hdf5_path = config['data']['val_hdf5_path'] # TODO: hau aldatzeko dago
    caption_config = config['data']['captions']
    keypoint_config = config['data']['keypoints']
    random_clip_config = config['data']['random_clips']
    model_name = config['model']['model_name']
    lora = config['model']['lora']
    save_path = config['model']['save_path']
    generation_max_length = config['generation']['max_length']
    max_length_config = config['training']['max_lengths']
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

    use_mt = (
        initial_data_proportions['mt'] != 0
        or proportions['mt'] != 0
    )
    if use_mt:
        print('Creating MT datasets...')
        mt_ds, reverse_mt_ds = load_mt_datasets(
            config["data"]["mt_pairs"],
            max_characters=caption_config["max_characters"]
        )
        len_mt = len(mt_ds)
    else:
        print('Skipping MT datasets because both MT data proportions are zero.')
        mt_ds = reverse_mt_ds = None
        len_mt = 0

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    references = get_bleu_references(val_dataset)

    def compute_metrics(eval_pred):
        predictions, _ = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Distributed evaluation pads the final global batch to a multiple of
        # the per-device batch size. In the currently installed
        # Transformers/Accelerate versions those extra examples can reach
        # compute_metrics, so remove them before computing corpus metrics.
        num_validation_samples = len(val_dataset)
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
        max_length_config=max_length_config,
        random_clip_config=random_clip_config
    )
    trainer_class = Seq2SeqTrainer
    trainer_extra_kwargs = {}
    if args.gradient_diagnostics:
        collator = GradientDiagnosticCollator(collator)
        trainer_class = GradientDiagnosticSeq2SeqTrainer
        trainer_extra_kwargs["gradient_spike_threshold"] = (
            args.gradient_spike_threshold
        )

    print('Preparing training data...')

    initial_mul_caption, _ = data_multipliers(initial_data_proportions, len_mt, len(train_caption_real), len(train_random_real))
    mul_caption, mul_random = data_multipliers(proportions, len_mt, len(train_caption_real), len(train_random_real))

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

    # Resumed runs restore model weights from a Trainer checkpoint. A fresh
    # second-phase-only run instead loads an initial-phase checkpoint below.
    load_weights = not args.resume_initial and not args.resume_second and not args.only_second

    model = CustomByT5Model(model_name=model_name, input_dim=input_dim, lora_config=lora_config, load_weights=load_weights)

    def create_optimizer():
        return Adafactor(
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
        seed=seed,
        data_seed=data_seed,
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
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2, # bestela bukaeran azkena borratzen du
        metric_for_best_model='eval_bleu',
        report_to='wandb',
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

        initial_training_args = dataclasses.replace(
            training_args,
            output_dir=os.path.join(save_path, args.run_name+'_initial'),
            run_name=args.run_name+'_initial',
            per_device_train_batch_size=initial_per_device_batch_sizes['train'],
            per_device_eval_batch_size=initial_per_device_batch_sizes['eval'],
            gradient_accumulation_steps=initial_gradient_accumulation_steps,
            max_steps=(
                args.gradient_diagnostics_max_steps
                if args.gradient_diagnostics_max_steps is not None
                else initial_max_steps
            ),
            eval_steps=2000,
            save_steps=2000,
        )

        initial_optimizer = create_optimizer()

        initial_trainer = trainer_class(
            model=model,
            args=initial_training_args,
            train_dataset=initial_train,
            eval_dataset=val_dataset, #dev_caption_real,
            data_collator=collator,
            optimizers=(initial_optimizer, None),
            compute_metrics=compute_metrics,
            **trainer_extra_kwargs,
        )

        initial_trainer.train(resume_from_checkpoint=args.resume_initial)

        if not args.only_initial and initial_trainer.is_world_process_zero() and 'wandb' in initial_training_args.report_to:
            import wandb
            wandb.finish()

    if args.only_second and not args.resume_second:
        if args.checkpoint_path is not None:
            checkpoint_path = args.checkpoint_path
        else:
            last_checkpoint = get_last_checkpoint(os.path.join(save_path, args.run_name+'_initial'))
            checkpoint_path = os.path.join(last_checkpoint, 'pytorch_model.bin')
        model.load_state_dict(torch.load(checkpoint_path))

    if not args.only_initial:

        # The second phase intentionally starts with fresh optimizer and
        # scheduler state, whether it follows the initial phase in this
        # execution or is launched separately with --only-second. When
        # --resume-second is used, Trainer restores the second-phase states
        # from its checkpoint into these newly created objects.
        if not args.resume_second:
            set_seed(seed)
        second_optimizer = create_optimizer()

        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train,
            eval_dataset=val_dataset, #{'caption': dev_caption_real, 'random': dev_random_real},
            data_collator=collator,
            optimizers=(second_optimizer, None),
            compute_metrics=compute_metrics,
            **trainer_extra_kwargs,
        )

        trainer.train(resume_from_checkpoint=args.resume_second)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a CustomByT5Model.')
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--config-path', type=str, default='configs/config.yaml')
    parser.add_argument('--only-initial', action='store_true')
    parser.add_argument('--only-second', action='store_true')
    parser.add_argument('--checkpoint-path', type=str)
    parser.add_argument('--resume-initial', action='store_true')
    parser.add_argument('--resume-second', action='store_true')
    parser.add_argument('--gradient-diagnostics', action='store_true')
    parser.add_argument('--gradient-spike-threshold', type=float, default=5.0)
    parser.add_argument('--gradient-diagnostics-max-steps', type=int)
    args = parser.parse_args()

    if args.only_initial and args.only_second:
        parser.error('--only-initial and --only-second cannot be used together')
    if args.resume_initial and args.only_second:
        parser.error('--resume-initial cannot be used with --only-second')
    if args.resume_second and not args.only_second:
        parser.error('--resume-second must be used with --only-second')
    if args.checkpoint_path is not None and not args.only_second:
        parser.error('--checkpoint-path must be used with --only-second')
    if args.checkpoint_path is not None and args.resume_second:
        parser.error('--checkpoint-path cannot be used with --resume-second')
    if (
        args.gradient_diagnostics_max_steps is not None
        and not args.gradient_diagnostics
    ):
        parser.error(
            '--gradient-diagnostics-max-steps requires '
            '--gradient-diagnostics'
        )

    print(args)

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    
    main(args, config)
