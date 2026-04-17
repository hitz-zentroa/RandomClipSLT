import argparse
import os
import torch
import yaml

from data.data import sentence_level_hdf5_to_dataset, DataCollator
from model import CustomByT5Model

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer



def finetune_how2sign(args, config):

    model_name = config['model']['model_name']
    input_dim = config['model']['input_dim']
    save_path = config['model']['save_path']
    # TODO: dataset desberdinetan doitzeko aukera ematean aldatu
    h5_path = os.path.join(
        config['data']['how2sign']['path'],
        'hdf5',
        'train.h5'
    )
    val_hdf5_path = config['data']['val_hdf5_path'] # TODO: hau ere aldatzeko
    max_tokens = config['training']['max_tokens']

    train_ds = sentence_level_hdf5_to_dataset(
        h5_path, 
        dataset_name='how2sign_train', 
        src_lang='ase', 
        tgt_lang='en'
    )

    print(f'Training dataset loaded from {h5_path} with {len(train_ds)} instances.')

    val_ds = sentence_level_hdf5_to_dataset(
        val_hdf5_path,
        dataset_name='how2sign_val',
        src_lang='ase',
        tgt_lang='en'
    )

    print(f'Validation dataset loaded from {val_hdf5_path} with {len(val_ds)} instances.')

    data_paths_dict = {
        'how2sign_train': h5_path,
        'how2sign_val': val_hdf5_path
    }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = DataCollator(
        data_paths_dict, 
        tokenizer, 
        input_dim, 
        max_tokens=max_tokens
    )

    model = CustomByT5Model(model_name=model_name, input_dim=input_dim, load_weights=False)
    model.load_state_dict(torch.load(args.checkpoint_path))

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(save_path, args.run_name),
        run_name=args.run_name,
        logging_steps=100, # default 500. Kendu?
        max_steps=5000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=1e-3, # BEGIRATU EA ALDATU BEHAR DEN
        optim="adafactor",
        #warmup_steps=1000,
        lr_scheduler_type='constant',
        eval_strategy='steps',
        save_strategy='steps',
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        metric_for_best_model='eval_loss',
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
        data_collator=collator
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
