import argparse
import h5py
import os
import torch
import yaml

from model import CustomByT5Model
from sacrebleu.metrics import BLEU
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from datasets import Dataset
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer



def decode_sentence(sentence):
    return bytes(list(sentence[(sentence != 0) & (sentence != 1)]-3)).decode('utf-8', errors='ignore')


def test_how2sign(model, tokenizer, batch_size, hdf5_path, num_beams=1, device='cpu'):

    ref = []
    out = []

    poses = []

    input_text = 'slt 1 src:ase tgt:en\n'
    encoded = tokenizer(batch_size * [input_text], padding=True, return_tensors='pt')
    input_ids = encoded['input_ids']
    text_attention_mask = encoded['attention_mask']

    input_ids = input_ids.to(device)
    text_attention_mask = text_attention_mask.to(device)

    with h5py.File(hdf5_path, 'r') as f:
        for _, sentence_group in f.items():
            pose = torch.tensor(sentence_group['processed_keypoints'][:])
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
                                        max_length=256)

        for output in outputs:
            decoded = decode_sentence(output)
            
            if '\n' in decoded:
                print('⚠️ Newline found in decoded output, replacing with space.')
                decoded = decoded.replace('\n', ' ')
            
            out.append(decoded)

    return ref, out


def main(args, config):

    model_name = config['model']['model_name']
    input_dim = config['model']['input_dim']
    lora = config['model']['lora']
    beam_size = config['generation']['beam_size']

    os.makedirs(args.output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomByT5Model(model_name=model_name, input_dim=input_dim, load_weights=False) # TODO: LoRA
    checkpoint_path = args.checkpoint_path
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    ref, out = test_how2sign(model, tokenizer, batch_size=64, hdf5_path=args.hdf5_path, num_beams=beam_size, device=device)
    
    bleu = BLEU()
    bleu_score = bleu.corpus_score(out, [ref])
    print(bleu_score)

    with open(os.path.join(args.output_path, 'ref.txt'), 'w') as f:
        f.write('\n'.join(ref) + '\n')

    with open(os.path.join(args.output_path, 'out.txt'), 'w') as f:
        f.write('\n'.join(out) + '\n')

    with open(os.path.join(args.output_path, 'bleu.txt'), 'w') as f:
        print(bleu_score, file=f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='configs/config.yaml')
    parser.add_argument('--output-path', type=str, default='outputs/default/')
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--hdf5-path', type=str, required=True)
    args = parser.parse_args()

    print(args)

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    print(config)

    main(args, config)


# predictions = trainer.predict(dev_dataset).predictions
# predictions[predictions == -100] = 0
# decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
# for sentence in decoded_preds:
#     print(sentence)
