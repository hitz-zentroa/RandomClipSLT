import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import cv2
import ffmpegio
import h5py
import pandas as pd
import mediapipe as mp
import numpy as np
import yaml

from extract_keypoints import process_frame, NUM_LANDMARKS

from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm



def process_video(dict_item, model_complexity):

    video_path, sentences = dict_item

    results = {}

    cap = cv2.VideoCapture(video_path)

    for sentence in sentences:

        cap.set(cv2.CAP_PROP_POS_MSEC, sentence['start_time'] * 1000)
        frame_array_list = []

        with mp.solutions.holistic.Holistic(
            static_image_mode=False, 
            model_complexity=model_complexity, 
            refine_face_landmarks=True
        ) as holistic:

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    break

                current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if current_time > sentence['end_time']:
                    break

                frame_vec = process_frame(frame, holistic)
                frame_array_list.append(frame_vec)

        if frame_array_list:
            video_array = np.stack(frame_array_list)
        else:
            video_array = np.empty((0, NUM_LANDMARKS, 3), dtype=np.float32)

        results[sentence['id']] = (video_array, sentence['sentence'])

    cap.release()

    probe = ffmpegio.probe.video_streams_basic(video_path)[0]

    return {
        'probe': probe,
        'results': results
    }



def main(args, config):

    model_complexity = config['data']['keypoints']['mediapipe_model_complexity']
    
    csv_path = os.path.join(
        config['data']['how2sign']['path'], 
        f'how2sign_realigned_{args.split}.csv'
    )
    
    video_dir = os.path.join(
        config['data']['how2sign']['path'], 
        f'{args.split}_raw_videos'
    )
    
    h5_path = os.path.join(
        config['data']['how2sign']['path'],
        'hdf5',
        f'{args.split}.h5'
    )

    data = pd.read_csv(csv_path, sep='\t')
    videos = defaultdict(list)
    for _, row in data.iterrows():
        video_path = os.path.join(video_dir, f'{row["VIDEO_NAME"]}.mp4')
        videos[video_path].append({
            'start_time': row['START_REALIGNED'],
            'end_time': row['END_REALIGNED'],
            'sentence': row['SENTENCE'],
            'id': row['SENTENCE_NAME']
        })

    sentence_count = sum(len(sentences) for sentences in videos.values())
    print(f'{sentence_count} sentences found.')

    partial_process_video = partial(process_video, model_complexity=model_complexity)

    with Pool(args.num_processes) as pool:
        for video_dict in tqdm(
            pool.imap_unordered(partial_process_video, videos.items()),
            total=len(videos),
            smoothing=0.1
        ):
            with h5py.File(h5_path, 'a') as f:
                for sentence_id, (video_array, sentence) in video_dict['results'].items():
                    sentence_group = f.require_group(sentence_id)
                    sentence_group.create_dataset('keypoints', data=video_array, compression='gzip', compression_opts=6)
                    sentence_group.attrs['sentence'] = sentence
                    sentence_group.attrs['fps'] = float(video_dict['probe']['frame_rate'])
                    sentence_group.attrs['width'] = video_dict['probe']['width']
                    sentence_group.attrs['height'] = video_dict['probe']['height']
            



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--config-path', type=str, default='configs/config.yaml')
    parser.add_argument('--num-processes', type=int, default=1)
    args = parser.parse_args()

    print(args)

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    
    main(args, config)
