import argparse
import cantofilter
import fasttext
import glob
import h5py
import langcodes
import os
import yaml

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .caption_utils import (
    caption_rows,
    create_caption_dataset,
    dataset_caption_rows,
    read_caption_file
)


def predict_lang(lang_model, lang_reg, text):
    reported_lang = lang_reg.split('-')[0]
    reported_lang = langcodes.get(reported_lang).language
    
    predicted_lang = lang_model.predict(text)[0][0].split('__')[-1]
    predicted_lang = langcodes.get(predicted_lang).language

    if predicted_lang == 'yue':
        if lang_reg in ['zh-HK', 'zh-TW', 'yue', 'yue-HK']:
            if cantofilter.judge(text) == 'mandarin':
                predicted_lang = 'zh-Hant'
            else:
                predicted_lang = 'yue'
        else:
            predicted_lang = reported_lang

    return predicted_lang


def process_caption_file(path, lang_model):

    lang_reg = os.path.basename(path).split('.')[1]

    captions = read_caption_file(path)

    all_text = ' '.join(caption['sentence'] for caption in captions).lower()
    predicted_lang = predict_lang(lang_model, lang_reg, all_text)

    return captions, predicted_lang


def choose_caption_file(video_id, predicted_lang, existing_filename, new_filename):
    prompt = (
        f'\nDifferent captions were found for video ID {video_id} in language '
        f'{predicted_lang}. Select which file to keep:\n'
        f'  1. {existing_filename}\n'
        f'  2. {new_filename}\n'
        'Enter 1 or 2: '
    )

    while True:
        choice = input(prompt).strip()
        if choice in {'1', '2'}:
            return choice == '2'
        print('Invalid selection. Enter 1 or 2.')


def main(args, config):

    lang_model_path = hf_hub_download(
        repo_id='facebook/fasttext-language-identification', 
        filename='model.bin'
    )
    
    lang_model = fasttext.load_model(lang_model_path)

    caps_glob = glob.glob(
        os.path.join(config['data']['captions']['path'], '*')
    )

    for caps_path in caps_glob:

        sign_lang = os.path.basename(caps_path)
        caps_file_paths = glob.glob(os.path.join(caps_path, '*.vtt'))
        print(f'Found {len(caps_file_paths)} caption files for sign language: {sign_lang}')

        hdf5_glob = glob.glob(
            os.path.join(config['data']['pretraining']['hdf5_path'], sign_lang, '*.h5')
        )

        if not hdf5_glob:
           print(f'HDF5 file not found for sign language: {sign_lang}')
           continue

        hdf5_path = hdf5_glob[0]

        rewritten_videos = set()
        chosen_filenames = {}

        with h5py.File(hdf5_path, 'a') as hdf5_file:
            for caps_file_path in tqdm(caps_file_paths):

                video_id = os.path.basename(caps_file_path).split('.')[0]

                if video_id not in hdf5_file:
                    print(f'Video ID {video_id} not found in HDF5 file for sign language: {sign_lang}')
                    continue

                captions, predicted_lang = process_caption_file(caps_file_path, lang_model)
                caps_filename = os.path.basename(caps_file_path)

                video_group = hdf5_file[video_id]
                # TODO: mugitu hau hemendik extract_keypoints.py-ra
                video_group.attrs['sign_lang'] = sign_lang

                if args.rewrite and video_id not in rewritten_videos:
                    if 'captions' in video_group:
                        if args.keep_chosen_files:
                            chosen_filenames[video_id] = {}
                            for lang, captions_ds in video_group['captions'].items():
                                filename = captions_ds.attrs.get('filename')
                                if isinstance(filename, bytes):
                                    filename = filename.decode('utf-8')
                                if filename is not None:
                                    chosen_filenames[video_id][lang] = filename
                        del video_group['captions']
                    rewritten_videos.add(video_id)

                captions_group = video_group.require_group('captions')

                chosen_filename = chosen_filenames.get(video_id, {}).get(predicted_lang)
                if chosen_filename is not None and caps_filename != chosen_filename:
                    continue

                if predicted_lang in captions_group:
                    existing_captions_ds = captions_group[predicted_lang]

                    if dataset_caption_rows(existing_captions_ds) == caption_rows(captions):
                        print(
                            f'Identical captions already exist for video ID {video_id} '
                            f'in language {predicted_lang}. Skipping.'
                        )
                        continue

                    existing_filename = existing_captions_ds.attrs.get(
                        'filename', '<unknown filename>'
                    )
                    if isinstance(existing_filename, bytes):
                        existing_filename = existing_filename.decode('utf-8')

                    replace_existing = choose_caption_file(
                        video_id,
                        predicted_lang,
                        existing_filename,
                        caps_filename
                    )
                    if not replace_existing:
                        continue

                    del captions_group[predicted_lang]

                create_caption_dataset(
                    captions_group,
                    predicted_lang,
                    captions,
                    filename=caps_filename
                )

        new_basename = os.path.basename(hdf5_path).replace('.h5', '_caps.h5')
        new_hdf5_path = os.path.join(config['data']['pretraining']['hdf5_path'], sign_lang, new_basename)
        os.rename(hdf5_path, new_hdf5_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess .vtt caption files.')
    parser.add_argument('--config-path', type=str, default='../configs/config.yaml')
    parser.add_argument('--rewrite', action='store_true')
    parser.add_argument('--keep-chosen-files', action='store_true')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    main(args, config)
