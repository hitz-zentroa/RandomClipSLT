import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import argparse
import glob
import h5py
import numpy as np
import os
import yaml

from extract_keypoints import OUT_OF_FRAME_NUM, NUM_HAND_LANDMARKS, NUM_MAIN_LANDMARKS, FACE_INDICES, NUM_LANDMARKS

from tqdm import tqdm


MAIN_FACE_INDICES = range(11)
MAIN_SHOULDER_INDICES = [11, 12]
MAIN_HANDS_INDICES = range(13, 23)
MAIN_HIPS_INDICES = [23, 24]
WRIST_INDEX = 0
NOSE_INDEX = FACE_INDICES.index(4)



def normalize_shoulder_dist(pose_data, main_landmarks_list, l_shoulder_index, r_shoulder_index, frame_freq=2):

    pose_data = pose_data[::frame_freq]

    T = pose_data.shape[0]

    l_hand = pose_data[:, :NUM_HAND_LANDMARKS]
    r_hand = pose_data[:, NUM_HAND_LANDMARKS:2*NUM_HAND_LANDMARKS]

    main_pose = pose_data[:, 2*NUM_HAND_LANDMARKS:
                             2*NUM_HAND_LANDMARKS + NUM_MAIN_LANDMARKS]

    face = pose_data[:, 2*NUM_HAND_LANDMARKS + NUM_MAIN_LANDMARKS:]

    main_pose = main_pose[:, main_landmarks_list]

    l_shoulder = main_pose[:, l_shoulder_index]
    r_shoulder = main_pose[:, r_shoulder_index]

    shoulders_valid = (
        (l_shoulder[:,0] != OUT_OF_FRAME_NUM) &
        (r_shoulder[:,0] != OUT_OF_FRAME_NUM)
    )

    neck = (l_shoulder + r_shoulder) / 2
    dist = np.linalg.norm(l_shoulder - r_shoulder, axis=1, keepdims=True)

    dist[dist < 1e-6] = 1


    # ----- normalize hands -----

    l_wrist = l_hand[:, WRIST_INDEX].copy()
    r_wrist = r_hand[:, WRIST_INDEX].copy()

    l_mask = l_hand[:,:,0] != OUT_OF_FRAME_NUM
    r_mask = r_hand[:,:,0] != OUT_OF_FRAME_NUM

    l_hand -= l_wrist[:,None,:]
    r_hand -= r_wrist[:,None,:]

    l_hand /= dist[:,None,:]
    r_hand /= dist[:,None,:]

    # restore wrist
    l_hand[:,WRIST_INDEX] = l_wrist
    r_hand[:,WRIST_INDEX] = r_wrist

    # restore OUT_OF_FRAME values
    l_hand[~l_mask] = OUT_OF_FRAME_NUM
    r_hand[~r_mask] = OUT_OF_FRAME_NUM

    # ----- normalize face -----

    nose = face[:, NOSE_INDEX].copy()

    face_mask = face[:, :, 0] != OUT_OF_FRAME_NUM

    face -= nose[:, None, :]
    face /= dist[:, None, :]

    # restore nose
    face[:, NOSE_INDEX] = nose

    # restore out-of-frame landmarks
    face[~face_mask] = OUT_OF_FRAME_NUM

    # ----- normalize main pose -----

    pose_mask = main_pose[:, :, 0] != OUT_OF_FRAME_NUM

    main_pose -= neck[:, None, :]
    main_pose /= dist[:, None, :]

    # restore OUT_OF_FRAME landmarks
    main_pose[~pose_mask] = OUT_OF_FRAME_NUM

    # ----- discard frames with invalid shoulders -----

    l_hand[~shoulders_valid] = OUT_OF_FRAME_NUM
    r_hand[~shoulders_valid] = OUT_OF_FRAME_NUM
    main_pose[~shoulders_valid] = OUT_OF_FRAME_NUM
    face[~shoulders_valid] = OUT_OF_FRAME_NUM

    frame_pose_array = np.concatenate(
        [l_hand, r_hand, main_pose, face],
        axis=1
    )

    return frame_pose_array.reshape(T, -1)


def select_pose_landmarks(pose_data, main_landmarks_list, frame_freq=2):

    selected_pose_data_list = []

    for frame_idx in range(0, pose_data.shape[0], frame_freq):
        
        frame_pose = pose_data[frame_idx]
        l_hand = frame_pose[0 : NUM_HAND_LANDMARKS]
        r_hand = frame_pose[NUM_HAND_LANDMARKS : 2*NUM_HAND_LANDMARKS]
        main_pose = frame_pose[2*NUM_HAND_LANDMARKS : 2*NUM_HAND_LANDMARKS+NUM_MAIN_LANDMARKS]
        face = frame_pose[2*NUM_HAND_LANDMARKS+NUM_MAIN_LANDMARKS : ]

        selected_main_pose = main_pose[main_landmarks_list]
        selected_frame_pose = {
            'l_hand': l_hand,
            'r_hand': r_hand,
            'main_pose': selected_main_pose,
            'face': face
        }

        selected_pose_data_list.append(selected_frame_pose)

    return selected_pose_data_list


def get_main_landmarks_list(main_landmarks_config):

    main_landmarks_list = []
    
    if main_landmarks_config['include_face']:
        main_landmarks_list += MAIN_FACE_INDICES
    
    main_landmarks_list += MAIN_SHOULDER_INDICES
    
    if main_landmarks_config['include_hands']:
        main_landmarks_list += MAIN_HANDS_INDICES

    if main_landmarks_config['include_hips']:
        main_landmarks_list += MAIN_HIPS_INDICES

    return main_landmarks_list


def main(args, config):

    frame_freq = config['data']['keypoints']['frame_freq']
    use_z = config['data']['keypoints']['use_z']
    main_landmarks_config = config['data']['keypoints']['main_pose_landmarks']
    frame_freq = config['data']['keypoints']['frame_freq']
    hdf5_path = config['data']['hdf5_path']

    main_landmarks_list = get_main_landmarks_list(main_landmarks_config)
    l_shoulder_index = main_landmarks_list.index(MAIN_SHOULDER_INDICES[0])
    r_shoulder_index = main_landmarks_list.index(MAIN_SHOULDER_INDICES[1])
    num_coords = 3 if use_z else 2

    data_paths = glob.glob(os.path.join(hdf5_path, '*', '*.h5'))
    sign_langs = [os.path.basename(os.path.dirname(path)) for path in data_paths]
    print(f'Found {len(data_paths)} HDF5 data files for sign languages: {", ".join(sign_langs)}')

    for data_path in data_paths:

        with h5py.File(data_path, 'a') as f:
            for _, video_group in tqdm(f.items()):

                pose_data = video_group['keypoints'][:]

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

                    inverse_aspect_ratio = video_group.attrs['height'] / video_group.attrs['width']

                    mask = pose_data[:, :, 1] != OUT_OF_FRAME_NUM
                    pose_data[:, :, 1][mask] *= inverse_aspect_ratio

                    # ---- changed part ----
                    normalized_pose_data = normalize_shoulder_dist(
                        pose_data,
                        main_landmarks_list,
                        l_shoulder_index,
                        r_shoulder_index,
                        frame_freq
                    )
                    # ----------------------

                if args.rewrite and 'processed_keypoints' in video_group:
                    del video_group['processed_keypoints']

                ds = video_group.create_dataset(
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

        # TODO: bug, honek ez du ondo funtzionatu
        new_basename = os.path.basename(hdf5_path).replace('.h5', '_caps.h5')
        dirname = os.path.dirname(data_path)
        new_hdf5_path = os.path.join(dirname, new_basename)
        os.rename(data_path, new_hdf5_path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='../configs/config.yaml')
    parser.add_argument('--rewrite', action='store_true')
    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    main(args, config)
