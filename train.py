import argparse
import pickle
import torch
import os

from data.data import load_ehealth_data, load_vtt_data, merge_pose_text_data, CustomDataCollator, PretrainingDataCollator
from model import CustomByT5Model

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer


#os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["WANDB__SERVICE_WAIT"] = "300" # ???

parser = argparse.ArgumentParser(description='Train a CustomByT5Model.')
parser.add_argument('--run-name', type=str, required=True)
parser.add_argument('--lora', action='store_true', default=False, help='Enable LoRA.')
args = parser.parse_args()

RUN_NAME = args.run_name

MODEL_NAME = 'google/byt5-small'
INPUT_DIM = 255

if args.lora:
    lora_config = LoraConfig(
        target_modules=['q', 'k', 'v', 'o'],
        modules_to_save=['lm_head']
    )
else:
    lora_config = None

model = CustomByT5Model(model_name=MODEL_NAME, input_dim=INPUT_DIM, lora_config=lora_config)

with open('/ikerlariak/anunez038/TRAIN/UVigo_features/mediapipe_features.pkl', 'rb') as f:
    pose_data = pickle.load(f)
text_data = load_ehealth_data('/ikerlariak/anunez038/TRAIN/LSE-Health-UVigo-Splits.xlsx')
train_data, dev_data, _ = merge_pose_text_data(pose_data,
                                               text_data,
                                               normalize_pose=True,
                                               video_info_path='/tartalo01/users/balkain001/TRAIN/YouTube-SL-25/UVigo_video_info.tsv')

# with open('/ikerlariak/anunez038/TRAIN/YouTube-SL-25/LSE_features/mediapipe_features.pkl', 'rb') as f:
#     yt_pose_data = pickle.load(f)
# yt_text_data = load_vtt_data('/ikerlariak/anunez038/TRAIN/YouTube-SL-25/LSE_720p30fps')
# yt_train_data, _, _ = merge_pose_text_data(yt_pose_data,
#                                            yt_text_data,
#                                            normalize_pose=True,
#                                            video_info_path='/tartalo01/users/balkain001/TRAIN/YouTube-SL-25/LSE_video_info.tsv')

# train_data += yt_train_data

train_dataset = Dataset.from_list(train_data)
dev_dataset = Dataset.from_list(dev_data)

tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')
collator = CustomDataCollator(tokenizer)


training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join('/gscratch5/users/balkain001/TRAIN/byt5', RUN_NAME),
    logging_steps=100,
    num_train_epochs=50,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1e-3, # hasierako esperimentuetan 2e-5
    weight_decay=0.01,
    predict_with_generate=False, # momentuz horrela epoka bakoitzeko ebaluazioan generate ez erabiltzeko (??)
    generation_max_length=256,
    save_strategy='epoch',
    eval_strategy='epoch',
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    report_to='wandb', # jarri gabe ere egiten du
    run_name=RUN_NAME,
    #prediction_loss_only=True,
    #fp16=True,
    #bf16=True, # biekin loss 0 ematen dit hasieratik
    remove_unused_columns=False, # https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76947480#76947480
    save_safetensors=False, # https://discuss.huggingface.co/t/saving-model-in-safetensors-format-through-trainer-fails-for-gemma-2-due-to-shared-tensors/109451
    dataloader_num_workers=4
) # beste batzuk zeuden hemen: https://medium.com/@anyuanay/fine-tuning-the-pre-trained-t5-small-model-in-hugging-face-for-text-summarization-3d48eb3c4360

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    data_collator=collator
)

trainer.train()

# Modeloa probatzeko (gero beste nonbaitera eraman):

#model.load_state_dict(torch.load('./results/checkpoint-1000/pytorch_model.bin'))

#eval_metrics = trainer.evaluate()
#print('eval_metrics:', eval_metrics)

# predictions = trainer.predict(dev_dataset)
# for sentence_pred in predictions.predictions:
#     print(sentence_pred)
#     print(tokenizer.decode(sentence_pred, skip_special_tokens=True))
    
#print(predictions.predictions[:10])
#decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
#print(decoded_preds)

