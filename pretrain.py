import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import dataclasses
import glob
import shutil
import torch
import yaml

from data.data import read_to_datasets, load_mt_datasets, sentence_level_hdf5_to_dataset, DataCollator
from model import CustomByT5Model

from datasets import concatenate_datasets
from peft import LoraConfig
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers.trainer_utils import get_last_checkpoint



def data_multipliers(proportions, len_mt, len_caption_clip, len_random_clip):
    
    # Asumitzen dut MT izango dela beti handiena
    
    aim_caption = proportions['caption_clip'] * len_mt * 2 / proportions['mt']
    mul_caption = round(aim_caption / len_caption_clip)

    aim_random = proportions['random_clip'] * len_mt * 2 / proportions['mt']
    mul_random = round(aim_random / len_random_clip)

    return mul_caption, mul_random


def main(args, config):

    os.environ["WANDB__SERVICE_WAIT"] = "300" # ???
    # TXAPUZATXOA MARTXAN ZEGOEN EXEKUZIOAREN JARRAIPENA IZAN DADIN
    #os.environ["WANDB_RESUME"] = 'must'
    #os.environ["WANDB_PROJECT"] = 'huggingface'
    #os.environ["WANDB_RUN_ID"] = '4q66kpy2'

    hdf5_path = config['data']['hdf5_path']
    val_hdf5_path = config['data']['val_hdf5_path'] # TODO: hau aldatzeko dago
    max_characters = config['data']['captions']['max_characters']
    max_seconds = config['data']['captions']['max_seconds']
    min_seconds = config['data']['captions']['min_seconds']
    captions_langs = config['data']['captions']['langs']
    frame_freq = config['data']['keypoints']['frame_freq']
    chunk_seconds = config['data']['random_clips']['chunk_seconds']
    min_duration_seconds = config['data']['random_clips']['min_duration_seconds']
    model_name = config['model']['model_name']
    input_dim = config['model']['input_dim']
    lora = config['model']['lora']
    save_path = config['model']['save_path']
    max_tokens = config['training']['max_tokens']
    initial_data_proportions = config['training']['initial_data_proportions']
    proportions = config['training']['data_proportions']

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
        data_paths,
        frame_freq=frame_freq,
        chunk_seconds=chunk_seconds,
        max_characters=max_characters,
        max_seconds=max_seconds,
        min_seconds=min_seconds,
        captions_langs=captions_langs
    )

    print(f'train_caption_real: {len(train_caption_real)} instances')
    print(f'dev_caption_real: {len(dev_caption_real)} instances')
    print(f'train_random_real: {len(train_random_real)} instances')
    print(f'dev_random_real: {len(dev_random_real)} instances')

    # TODO: config-etik hartu zein izango den dataseta eta hortik hartu ezaugarriak
    val_dataset = sentence_level_hdf5_to_dataset(
        val_hdf5_path, 
        dataset_name='how2sign_val', 
        src_lang='ase', 
        tgt_lang='en'
    )
    data_paths_dict['how2sign_val'] = val_hdf5_path
    print(f'Validation dataset loaded from {val_hdf5_path} with {len(val_dataset)} instances.')

    print('Creating MT datasets...')
    
    mt_ds, reverse_mt_ds = load_mt_datasets(config['data']['mt_pairs'], max_characters=max_characters)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    collator = DataCollator(
        data_paths_dict, 
        tokenizer, 
        input_dim, 
        chunk_seconds=chunk_seconds, 
        min_duration_seconds=min_duration_seconds,
        max_tokens=max_tokens
    )

    print('Preparing training data...')

    initial_mul_caption, _ = data_multipliers(initial_data_proportions, len(mt_ds), len(train_caption_real), len(train_random_real))
    mul_caption, mul_random = data_multipliers(proportions, len(mt_ds), len(train_caption_real), len(train_random_real))

    initial_oversampled_train_caption_real = concatenate_datasets([train_caption_real] * initial_mul_caption)
    initial_train = concatenate_datasets([initial_oversampled_train_caption_real, mt_ds, reverse_mt_ds])

    oversampled_train_caption_real = concatenate_datasets([train_caption_real] * mul_caption)
    oversampled_train_random_real = concatenate_datasets([train_random_real] * mul_random)
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

    training_args = Seq2SeqTrainingArguments(
        os.path.join(save_path, args.run_name),
        logging_steps=100,
        max_steps=300_000,
        per_device_train_batch_size=16, # 8rekin lehen 53.100. pausoan out of memory
        per_device_eval_batch_size=64,
        learning_rate=1e-3, # hasierako esperimentuetan 2e-5
        optim='adafactor',
        warmup_steps=1000,
        lr_scheduler_type='constant',
        #weight_decay=0.01,
        predict_with_generate=False, # momentuz horrela epoka bakoitzeko ebaluazioan generate ez erabiltzeko (??)
        generation_max_length=256,
        eval_strategy='steps',
        save_strategy='steps',
        eval_steps=5000,
        save_steps=5000,
        save_total_limit=2, # bestela bukaeran azkena borratzen du
        metric_for_best_model='eval_loss', # 'eval_caption_loss',
        #report_to='wandb', # jarri gabe ere egiten du
        run_name=args.run_name,
        #prediction_loss_only=True,
        #fp16=True,
        bf16=True,
        remove_unused_columns=False, # https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76947480#76947480
        save_safetensors=False, # https://discuss.huggingface.co/t/saving-model-in-safetensors-format-through-trainer-fails-for-gemma-2-due-to-shared-tensors/109451
        dataloader_num_workers=16
    ) # beste batzuk zeuden hemen: https://medium.com/@anyuanay/fine-tuning-the-pre-trained-t5-small-model-in-hugging-face-for-text-summarization-3d48eb3c4360

    if not args.only_second:

        initial_training_args = dataclasses.replace(
            training_args,
            output_dir=os.path.join(save_path, args.run_name+'_initial'),
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            max_steps=300_000, #100_000,
            # metric_for_best_model='eval_loss',
        )

        initial_trainer = Seq2SeqTrainer(
            model=model,
            args=initial_training_args,
            train_dataset=initial_train,
            eval_dataset=val_dataset, #dev_caption_real,
            data_collator=collator
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
            data_collator=collator
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
