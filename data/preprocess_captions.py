import argparse
import cantofilter
import fasttext
import glob
import h5py
import langcodes
import numpy as np
import os
import webvtt
import yaml

from bs4 import BeautifulSoup
from huggingface_hub import hf_hub_download
from tqdm import tqdm



def clean_caption_text(caption_text):
    return BeautifulSoup((BeautifulSoup(caption_text, 'lxml').text), 'lxml').text.replace('\n', ' ')


def get_caption_time_ms(caption_time):
    h, min, s, ms = caption_time.to_tuple()
    return h * 3_600_000 + min * 60_000 + s * 1000 + ms


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
    
    captions = []

    for caption in webvtt.read(path):
        
        clean_text = clean_caption_text(caption.text)
        start_time = get_caption_time_ms(caption.start_time)
        end_time = get_caption_time_ms(caption.end_time)

        captions.append({
                'sentence': clean_text,
                'start_time': start_time,
                'end_time': end_time
            })

    all_text = ' '.join(caption['sentence'] for caption in captions).lower()
    predicted_lang = predict_lang(lang_model, lang_reg, all_text)

    return captions, predicted_lang


def main(args, config):

    lang_model_path = hf_hub_download(
        repo_id='facebook/fasttext-language-identification', 
        filename='model.bin'
    )
    
    lang_model = fasttext.load_model(lang_model_path)

    caption_dtype = np.dtype([
        ('sentence', h5py.string_dtype(encoding='utf-8')),
        ('start_time', np.int32),
        ('end_time', np.int32)
    ])

    caps_glob = glob.glob(
        os.path.join(config['data']['captions']['path'], '*')
    )

    for caps_path in caps_glob:

        sign_lang = os.path.basename(caps_path)
        caps_file_paths = glob.glob(os.path.join(caps_path, '*.vtt'))
        print(f'Found {len(caps_file_paths)} caption files for sign language: {sign_lang}')

        hdf5_glob = glob.glob(
            os.path.join(config['data']['hdf5_path'], sign_lang, '*.h5')
        )

        if not hdf5_glob:
           print(f'HDF5 file not found for sign language: {sign_lang}')
           continue

        hdf5_path = hdf5_glob[0]

        rewritten_videos = set()

        with h5py.File(hdf5_path, 'a') as hdf5_file:
            for caps_file_path in tqdm(caps_file_paths):

                video_id = os.path.basename(caps_file_path).split('.')[0]

                if video_id not in hdf5_file:
                    print(f'Video ID {video_id} not found in HDF5 file for sign language: {sign_lang}')
                    continue

                captions, predicted_lang = process_caption_file(caps_file_path, lang_model)

                video_group = hdf5_file[video_id]
                # TODO: mugitu hau hemendik extract_keypoints.py-ra
                video_group.attrs['sign_lang'] = sign_lang

                if args.rewrite and video_id not in rewritten_videos:
                    if 'captions' in video_group:
                        del video_group['captions']
                    rewritten_videos.add(video_id)

                captions_group = video_group.require_group('captions')

                if predicted_lang in captions_group:
                    print(f'Captions already exist for video ID {video_id} in language {predicted_lang}. Skipping.')
                    continue

                captions_array = np.array([(
                    caption['sentence'], 
                    caption['start_time'], 
                    caption['end_time']) for caption in captions], dtype=caption_dtype)
                
                captions_ds = captions_group.create_dataset(predicted_lang, data=captions_array)
                captions_ds.attrs['is_synthetic'] = False

        new_basename = os.path.basename(hdf5_path).replace('.h5', '_caps.h5')
        new_hdf5_path = os.path.join(config['data']['hdf5_path'], sign_lang, new_basename)
        os.rename(hdf5_path, new_hdf5_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Preprocess .vtt caption files.')
    parser.add_argument('--config-path', type=str, default='../configs/config.yaml')
    parser.add_argument('--rewrite', action='store_true')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    main(args, config)
