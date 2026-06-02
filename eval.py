import argparse
import h5py
import os
import torch
import yaml

from data.keypoint_processing import (
    get_keypoint_processing_config,
    get_processed_keypoint_dim,
    process_keypoint_clip,
)
from model import CustomByT5Model


from sacrebleu.metrics import BLEU
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from datasets import Dataset
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer


# TODO: eval_utils.py-ra eraman dudana hemendik kendu

def decode_sentence(sentence):
    return bytes(list(sentence[(sentence != 0) & (sentence != 1)]-3)).decode('utf-8', errors='ignore')


def decode_outputs(outputs, model_name, tokenizer):
    use_byt5_decoder = 'byt5' in model_name.lower()

    if use_byt5_decoder:
        decoded_batch = [decode_sentence(output) for output in outputs]
    else:
        decoded_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    cleaned_batch = []
    for decoded in decoded_batch:
        if '\n' in decoded:
            print('⚠️ Newline found in decoded output, replacing with space.')
            decoded = decoded.replace('\n', ' ')
        cleaned_batch.append(decoded)

    return cleaned_batch


def update_generation_stats(stats, outputs, tokenizer, generation_max_length):
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    for output in outputs:
        output_list = output.tolist()

        if eos_token_id in output_list:
            has_eos = True
            effective_length = output_list.index(eos_token_id) + 1
        else:
            has_eos = False
            effective_length = len(output_list)
            if pad_token_id is not None:
                while effective_length > 0 and output_list[effective_length - 1] == pad_token_id:
                    effective_length -= 1

        stats['total'] += 1
        stats['hit_max_length'] += int(effective_length == generation_max_length)
        stats['no_eos'] += int(not has_eos)
        stats['effective_length_sum'] += effective_length


def format_generation_stats(stats, generation_max_length):
    total = stats['total']
    if total == 0:
        return (
            f'Generation max_length: {generation_max_length}\n'
            'Generated outputs: 0\n'
            'Hit max_length: 0 (0.00%)\n'
            'No EOS: 0 (0.00%)\n'
            'Mean generated token length: 0.00'
        )

    hit_max_length_pct = stats['hit_max_length'] / total * 100
    no_eos_pct = stats['no_eos'] / total * 100
    mean_effective_length = stats['effective_length_sum'] / total

    return (
        f'Generation max_length: {generation_max_length}\n'
        f'Generated outputs: {total}\n'
        f'Hit max_length: {stats["hit_max_length"]} ({hit_max_length_pct:.2f}%)\n'
        f'No EOS: {stats["no_eos"]} ({no_eos_pct:.2f}%)\n'
        f'Mean generated token length: {mean_effective_length:.2f}'
    )


def test_how2sign(
    model,
    model_name,
    tokenizer,
    batch_size,
    hdf5_path,
    keypoint_processing_config,
    num_beams=1,
    generation_max_length=128,
    device='cpu'
):

    ref = []
    out = []
    generation_stats = {
        'total': 0,
        'hit_max_length': 0,
        'no_eos': 0,
        'effective_length_sum': 0
    }

    poses = []

    input_text = 'slt 1 src:ase tgt:en\n'
    encoded = tokenizer(
        batch_size * [input_text],
        padding=True,
        return_tensors='pt',
        add_special_tokens=False
    )
    input_ids = encoded['input_ids']
    text_attention_mask = encoded['attention_mask']

    input_ids = input_ids.to(device)
    text_attention_mask = text_attention_mask.to(device)

    with h5py.File(hdf5_path, 'r') as f:
        for _, sentence_group in f.items():
            pose = torch.tensor(
                process_keypoint_clip(
                    sentence_group['keypoints'][:],
                    sentence_group.attrs['width'],
                    sentence_group.attrs['height'],
                    keypoint_processing_config
                ),
                dtype=torch.float32
            )
            
            # PROBA HONETARAKO BAKARRIK:
            # pose[pose == -100] = -5

            poses.append(pose)
            sentence = sentence_group.attrs['sentence']
            ref.append(sentence)

    for i in tqdm(range(0, len(poses), batch_size)):
        batch_poses = poses[i:i+batch_size]
        padded_poses = pad_sequence(batch_poses, batch_first=True, padding_value=0.0)
        pose_attention_mask = (~torch.all(padded_poses == 0.0, dim=-1)).long()

        padded_poses = padded_poses.to(device)
        pose_attention_mask = pose_attention_mask.to(device)

        real_size = len(batch_poses)

        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.generate(input_ids=input_ids[:real_size], 
                                        input_vectors=padded_poses, 
                                        text_attention_mask=text_attention_mask[:real_size],
                                        vectors_attention_mask=pose_attention_mask,
                                        num_beams=num_beams,
                                        #repetition_penalty=1.5,
                                        #length_penalty=0.6,
                                        max_length=generation_max_length)

        update_generation_stats(generation_stats, outputs, tokenizer, generation_max_length)
        out.extend(decode_outputs(outputs, model_name, tokenizer))

    return ref, out, generation_stats


def main(args, config):

    model_name = config['model']['model_name']
    lora = config['model']['lora']
    keypoint_config = config['data']['keypoints']
    beam_size = config['generation']['beam_size']
    generation_max_length = config['generation']['max_length']
    h5_path = os.path.join(
        config['data']['how2sign']['path'],
        'hdf5',
        f'{args.split}.h5'
    )

    os.makedirs(args.output_path)

    input_dim = get_processed_keypoint_dim(keypoint_config)
    print('input_dim:', input_dim)
    keypoint_processing_config = get_keypoint_processing_config(keypoint_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomByT5Model(model_name=model_name, input_dim=input_dim, load_weights=False) # TODO: LoRA
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    ref, out, generation_stats = test_how2sign(
        model,
        model_name,
        tokenizer,
        batch_size=64,
        hdf5_path=h5_path,
        keypoint_processing_config=keypoint_processing_config,
        num_beams=beam_size,
        device=device,
        generation_max_length=generation_max_length
    )
    
    bleu = BLEU()
    bleu_score = bleu.corpus_score(out, [ref])
    generation_stats_text = format_generation_stats(generation_stats, generation_max_length)
    print(bleu_score)
    print(generation_stats_text)

    with open(os.path.join(args.output_path, 'ref.txt'), 'w') as f:
        f.write('\n'.join(ref) + '\n')

    with open(os.path.join(args.output_path, 'out.txt'), 'w') as f:
        f.write('\n'.join(out) + '\n')

    with open(os.path.join(args.output_path, 'bleu.txt'), 'w') as f:
        print(bleu_score, file=f)
        print(generation_stats_text, file=f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='configs/config.yaml')
    parser.add_argument('--output-path', type=str, default='outputs/default/')
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True, choices=['val', 'test'])
    args = parser.parse_args()

    print(args)

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    print(config)

    main(args, config)

# TODO: noizbait probatu

# predictions = trainer.predict(dev_dataset).predictions
# predictions[predictions == -100] = 0
# decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
# for sentence in decoded_preds:
#     print(sentence)
