import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import glob
import h5py
import numpy as np
import os
import yaml

from extract_keypoints import OUT_OF_FRAME_NUM, NUM_HAND_LANDMARKS, NUM_MAIN_LANDMARKS, FACE_INDICES, NUM_LANDMARKS
from preprocess_keypoints_gpt import MAIN_FACE_INDICES, MAIN_SHOULDER_INDICES, MAIN_HANDS_INDICES, MAIN_HIPS_INDICES, WRIST_INDEX, NOSE_INDEX, normalize_shoulder_dist, get_main_landmarks_list

from tqdm import tqdm



def main(args, config):

    frame_freq = config['data']['keypoints']['frame_freq']
    use_z = config['data']['keypoints']['use_z']
    main_landmarks_config = config['data']['keypoints']['main_pose_landmarks']
    frame_freq = config['data']['keypoints']['frame_freq']
    h5_path = os.path.join(
        config['data']['how2sign']['path'],
        'hdf5',
        f'{args.split}.h5'
    )

    main_landmarks_list = get_main_landmarks_list(main_landmarks_config)
    l_shoulder_index = main_landmarks_list.index(MAIN_SHOULDER_INDICES[0])
    r_shoulder_index = main_landmarks_list.index(MAIN_SHOULDER_INDICES[1])
    num_coords = 3 if use_z else 2

    with h5py.File(h5_path, 'a') as f:
        for sentence_group in tqdm(f.values()):

            pose_data = sentence_group['keypoints'][:]

            assert pose_data.shape[1] == NUM_LANDMARKS, \
                    f'Unexpected number of keypoints: {pose_data.shape[1]}'

            if pose_data.shape[0] == 0:
                
                normalized_pose_data = np.empty(
                    (0, num_coords * (2*NUM_HAND_LANDMARKS +
                    len(main_landmarks_list) + len(FACE_INDICES))),
                    dtype=np.float32
                )
            
            else:

                if not use_z:
                    pose_data = pose_data[:, :, :2]

                inverse_aspect_ratio = sentence_group.attrs['height'] / sentence_group.attrs['width']

                mask = pose_data[:, :, 1] != OUT_OF_FRAME_NUM
                pose_data[:, :, 1][mask] *= inverse_aspect_ratio

                normalized_pose_data = normalize_shoulder_dist(
                    pose_data,
                    main_landmarks_list,
                    l_shoulder_index,
                    r_shoulder_index,
                    frame_freq
                )

            if args.rewrite and 'processed_keypoints' in sentence_group:
                del sentence_group['processed_keypoints']

            ds = sentence_group.create_dataset(
                'processed_keypoints',
                data=normalized_pose_data,
                compression='gzip',
                compression_opts=6
            )

            ds.attrs['description'] = (
                f'norm_shoulder_dist, hand_centered, face_centered, '
                f'frame_freq={frame_freq}, use_z={use_z}, '
                f'include_face={main_landmarks_config["include_face"]}, '
                f'include_hands={main_landmarks_config["include_hands"]}, '
                f'include_hips={main_landmarks_config["include_hips"]}'
            )



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--config-path', type=str, default='../configs/config.yaml')
    parser.add_argument('--rewrite', action='store_true')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    main(args, config)
