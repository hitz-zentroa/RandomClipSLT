import argparse
import os
import torch
import yaml

from data.data import T5_NEWLINE_TOKEN, sentence_level_hdf5_to_dataset, DataCollator
from data.keypoint_processing import get_processed_keypoint_dim
from eval_utils import compute_bleu_from_predictions, compute_generation_metrics, get_bleu_references
from model import CustomByT5Model

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, Adafactor, set_seed


def finetune_how2sign(args, config):

    seed = config['training'].get('seed', 42)
    set_seed(seed)
    print(f'Random seed: {seed}')

    model_name = config['model']['model_name']
    save_path = config['model']['save_path']
    # TODO: dataset desberdinetan doitzeko aukera ematean aldatu
    h5_path = os.path.join(
        config['data']['how2sign']['path'],
        'hdf5',
        'train.h5'
    )
    val_hdf5_path = config['data']['val_hdf5_path'] # TODO: hau ere aldatzeko
    keypoint_config = config['data']['keypoints']
    generation_max_length = config['generation']['max_length']
    max_length_config = config['training']['max_lengths']
    target_effective_batch_size = config['training']['finetune_target_effective_batch_size']
    per_device_batch_sizes = config['training']['finetune_per_device_batch_sizes']
    lr = config['training']['finetune_lr']
    max_steps = config['training']['finetune_max_steps']

    train_ds = sentence_level_hdf5_to_dataset(
        h5_path, 
        dataset_name='how2sign_train', 
        src_lang='ase', 
        tgt_lang='en'
    )

    print(f'Training dataset loaded with {len(train_ds)} instances.')

    val_ds = sentence_level_hdf5_to_dataset(
        val_hdf5_path,
        dataset_name='how2sign_val',
        src_lang='ase',
        tgt_lang='en'
    )

    print(f'Validation dataset loaded with {len(val_ds)} instances.')

    data_paths_dict = {
        'how2sign_train': h5_path,
        'how2sign_val': val_hdf5_path
    }

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
    model.load_state_dict(torch.load(args.checkpoint_path))

    optimizer = Adafactor(
        model.parameters(),
        lr=lr,
        relative_step=False,
        scale_parameter=False
    )

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    gradient_accumulation_steps = target_effective_batch_size // (per_device_batch_sizes['train'] * world_size)
    print(f'Gradient accumulation steps: {gradient_accumulation_steps} (target effective batch size: {target_effective_batch_size}, per device batch size: {per_device_batch_sizes["train"]}, world size: {world_size})')

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(save_path, args.run_name),
        run_name=args.run_name,
        seed=seed,
        data_seed=seed,
        logging_steps=50, # default 500. Kendu?
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
        save_strategy='steps',
        eval_steps=150,
        save_steps=150,
        save_total_limit=2,
        metric_for_best_model='eval_bleu',
        predict_with_generate=True,
        generation_max_length=generation_max_length,
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
    args = parser.parse_args()

    print(args)

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    print(config)

    finetune_how2sign(args, config)
