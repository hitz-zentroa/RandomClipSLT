import numpy as np

from data.keypoint_schema import (
    OUT_OF_FRAME_NUM,
    NUM_HAND_LANDMARKS,
    NUM_MAIN_LANDMARKS,
    FACE_INDICES,
    NUM_LANDMARKS,
    MAIN_FACE_INDICES,
    MAIN_SHOULDER_INDICES,
    MAIN_ELBOW_INDICES,
    MAIN_HANDS_INDICES,
    MAIN_HIPS_INDICES,
    WRIST_INDEX,
    NOSE_INDEX,
)


def get_main_landmarks_list(main_landmarks_config):

    main_landmarks_list = []
    
    if main_landmarks_config['include_face']:
        main_landmarks_list += MAIN_FACE_INDICES
    
    main_landmarks_list += MAIN_SHOULDER_INDICES
    main_landmarks_list += MAIN_ELBOW_INDICES
    
    if main_landmarks_config['include_hands']:
        main_landmarks_list += MAIN_HANDS_INDICES

    if main_landmarks_config['include_hips']:
        main_landmarks_list += MAIN_HIPS_INDICES

    return main_landmarks_list


def get_processed_keypoint_dim(keypoint_config, main_landmarks_list=None):
    if main_landmarks_list is None:
        main_landmarks_list = get_main_landmarks_list(
            keypoint_config["main_pose_landmarks"]
        )

    num_coords = 3 if keypoint_config["use_z"] else 2

    return num_coords * (
        2 * NUM_HAND_LANDMARKS
        + len(main_landmarks_list)
        + len(FACE_INDICES)
    )


def get_keypoint_processing_config(keypoint_config):
    main_landmarks_list = get_main_landmarks_list(
        keypoint_config["main_pose_landmarks"]
    )

    return {
        "frame_freq": keypoint_config["frame_freq"],
        "use_z": keypoint_config["use_z"],
        "main_landmarks_list": main_landmarks_list,
        "l_shoulder_index": main_landmarks_list.index(MAIN_SHOULDER_INDICES[0]),
        "r_shoulder_index": main_landmarks_list.index(MAIN_SHOULDER_INDICES[1]),
        "output_dim": get_processed_keypoint_dim(
            keypoint_config,
            main_landmarks_list=main_landmarks_list
        ),
    }


def normalize_shoulder_dist(pose_data, main_landmarks_list, l_shoulder_index, r_shoulder_index):
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
    dist = np.linalg.norm(
        l_shoulder[:, :2] - r_shoulder[:, :2],
        axis=1,
        keepdims=True
    )

    dist[dist < 1e-6] = 1
    
    # medianarekin egiteko. Ez zirudien lagungarria:
    # valid_dists = dist[shoulders_valid]
    # median_dist = float(np.median(valid_dists)) if len(valid_dists) > 0 and np.median(valid_dists) > 1e-6 else 1.0
    # dist = np.full_like(dist, median_dist)

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

    l_wrist_valid = l_wrist[:, 0] != OUT_OF_FRAME_NUM
    r_wrist_valid = r_wrist[:, 0] != OUT_OF_FRAME_NUM
    l_hand[l_wrist_valid, WRIST_INDEX] = (
        l_wrist[l_wrist_valid] - neck[l_wrist_valid]
    ) / dist[l_wrist_valid]
    r_hand[r_wrist_valid, WRIST_INDEX] = (
        r_wrist[r_wrist_valid] - neck[r_wrist_valid]
    ) / dist[r_wrist_valid]

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

    nose_valid = nose[:, 0] != OUT_OF_FRAME_NUM
    face[nose_valid, NOSE_INDEX] = (
        nose[nose_valid] - neck[nose_valid]
    ) / dist[nose_valid]

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


def process_keypoint_clip(pose_data, width, height, keypoint_processing_config):
    frame_freq = keypoint_processing_config["frame_freq"]
    use_z = keypoint_processing_config["use_z"]
    main_landmarks_list = keypoint_processing_config["main_landmarks_list"]
    l_shoulder_index = keypoint_processing_config["l_shoulder_index"]
    r_shoulder_index = keypoint_processing_config["r_shoulder_index"]

    assert pose_data.shape[1] == NUM_LANDMARKS, \
        f'Unexpected number of landmarks. Got {pose_data.shape[1]}, expected {NUM_LANDMARKS}.'

    pose_data = pose_data[::frame_freq]

    if pose_data.shape[0] == 0:
        return np.empty((0, keypoint_processing_config["output_dim"]), dtype=np.float32)

    if not use_z:
        pose_data = pose_data[:, :, :2]

    inverse_aspect_ratio = height / width

    mask = pose_data[:, :, 1] != OUT_OF_FRAME_NUM
    pose_data[:, :, 1][mask] *= inverse_aspect_ratio

    pose_data = normalize_shoulder_dist(
        pose_data,
        main_landmarks_list,
        l_shoulder_index,
        r_shoulder_index
    )

    return pose_data
