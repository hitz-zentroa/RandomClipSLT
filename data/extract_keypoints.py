import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["GLOG_minloglevel"] = "2" # TODO: ez du funtzionatu, berdin idazten du
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # TODO: ez du funtzionatu, berdin idazten du

import argparse
import cv2
import ffmpegio
import glob
import h5py
import mediapipe as mp
import numpy as np
import time
import yaml

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm


OUT_OF_FRAME_NUM = -100
NUM_HAND_LANDMARKS = 21
NUM_MAIN_LANDMARKS = 25 # legs are always discarded
FACE_INDICES = [0, 4, 13, 14, 17, 33, 37, 39, 46, 52, 55, 61, 64, 81, 82, 93, 133, 151, 152, 159, 172, 178, 181, 263, 269, 276, 282, 285, 291, 294, 311, 323, 362, 386, 397, 468, 473]
NUM_LANDMARKS = 2*NUM_HAND_LANDMARKS + NUM_MAIN_LANDMARKS + len(FACE_INDICES)



def process_frame(frame, holistic):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    frame_array = []

    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            frame_array.append([landmark.x, landmark.y, landmark.z])
    else:
        for _ in range(NUM_HAND_LANDMARKS):
            frame_array.append(3*[OUT_OF_FRAME_NUM])

    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            frame_array.append([landmark.x, landmark.y, landmark.z])
    else:
        for _ in range(NUM_HAND_LANDMARKS):
            frame_array.append(3*[OUT_OF_FRAME_NUM])

    if results.pose_landmarks:
        for index in range(NUM_MAIN_LANDMARKS):
            landmark = results.pose_landmarks.landmark[index]
            frame_array.append([landmark.x, landmark.y, landmark.z])
    else:
        for _ in range(NUM_MAIN_LANDMARKS):
            frame_array.append(3*[OUT_OF_FRAME_NUM])

    if results.face_landmarks:
        for index in FACE_INDICES:
            landmark = results.face_landmarks.landmark[index]
            frame_array.append([landmark.x, landmark.y, landmark.z])
    else:
        for _ in range(len(FACE_INDICES)):
            frame_array.append(3*[OUT_OF_FRAME_NUM])
    
    return np.array(frame_array, dtype=np.float32)


def video_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def process_video(video_path, model_complexity):
    
    id = video_id(video_path)
    print(f'Processing video {id}', flush=True)

    cap = cv2.VideoCapture(video_path)
    frame_array_list = []
    start_time = time.time()
    
    with mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=model_complexity, refine_face_landmarks=True) as holistic:

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            frame_vec = process_frame(frame, holistic)
            frame_array_list.append(frame_vec)

    cap.release()

    if frame_array_list:
        video_array = np.stack(frame_array_list)
    else:
        video_array = np.empty((0, NUM_LANDMARKS, 3), dtype=np.float32)

    probe = ffmpegio.probe.video_streams_basic(video_path)[0]

    processing_time = time.time() - start_time
    print(f'Video {id} processed in {processing_time:.1f} s. {len(frame_array_list)/processing_time:.1f} frames/s.', flush=True)

    return {
        'id': id,
        'video_array': video_array,
        'probe': probe
    }


def main(args, config):

    model_complexity = config['data']['keypoints']['mediapipe_model_complexity']

    videos_glob = glob.glob(
        os.path.join(config['data']['videos']['path'], '*')
    )

    for videos_path in videos_glob:

        sign_lang = os.path.basename(videos_path)
        video_paths = (
            glob.glob(os.path.join(videos_path, '*.mp4')) 
            + glob.glob(os.path.join(videos_path, '*.webm'))
        )
        
        print(f'Found {len(video_paths)} video files for sign language: {sign_lang}', flush=True)

        sign_lang_path = os.path.join(config['data']['hdf5_path'], sign_lang)
        h5_path = os.path.join(sign_lang_path, 'data.h5')

        if os.path.exists(h5_path):

            existing_ids = set()

            with h5py.File(h5_path, 'r') as f:
                for id, video_group in f.items():
                    if isinstance(video_group, h5py.Group) and 'keypoints' in video_group:
                        existing_ids.add(id)

            video_paths = [video_path for video_path in video_paths if video_id(video_path) not in existing_ids]
            print(f'Resuming. Found keypoints for {len(existing_ids)} videos for sign language: {sign_lang}. Remaining videos: {len(video_paths)}', flush=True)

        if not os.path.exists(sign_lang_path):
            os.makedirs(sign_lang_path)

        partial_process_video = partial(process_video, model_complexity=model_complexity)

        with Pool(args.num_processes) as pool:
            for result in tqdm(
                pool.imap_unordered(partial_process_video, video_paths),
                total=len(video_paths),
                smoothing=0.1
            ):
                with h5py.File(h5_path, 'a') as f:
                    video_group = f.require_group(result['id'])
                    video_group.create_dataset('keypoints', data=result['video_array'], compression='gzip', compression_opts=6)
                    video_group.attrs['fps'] = float(result['probe']['frame_rate'])
                    video_group.attrs['width'] = result['probe']['width']
                    video_group.attrs['height'] = result['probe']['height']
            


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='configs/config.yaml')
    parser.add_argument('--num-processes', type=int, default=1)
    args = parser.parse_args()

    print(args)

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    print(config)
    
    main(args, config)
