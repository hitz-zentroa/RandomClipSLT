import h5py
import numpy as np
import pandas as pd
import random
import torch

from datasets import Dataset, concatenate_datasets, load_dataset
from more_itertools import chunked_even
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm



# start_time and end_time are in ms
def extract_interval_from_sequence(vectors, start_time, end_time, sampled_fps=12.5):
    start_frame = round(start_time / 1000 * sampled_fps)
    end_frame = round(end_time / 1000 * sampled_fps)
    return vectors[start_frame:end_frame]


def load_ehealth_data(excel_path):

    data = []

    df_splits = pd.read_excel(
        excel_path, 
        sheet_name='#Segments',
        usecols=[1, 3, 4, 5],
        skiprows=[1],
        skipfooter=6,
        index_col=0
    )

    df_content = pd.read_excel(
        excel_path, 
        sheet_name='SegmentsContent',
        usecols=[0, 1, 2, 4],
        skipfooter=6
    )

    for _, row in df_content.iterrows():

        file = row['File']

        data.append({
            'file': file,
            'split': df_splits.loc[file]['Split SD&TD'],
            'sentence': row['PlainText: [S: What it was signed'], # Ez dut minuskulaz jartzen eta ez diot inolako garbitekarik aplikatzen
            'start_time': row['Start(ms)'] / 1000,
            'end_time': row['End(ms)'] / 1000
        })

    return data


# def simple_normalize(pose_data, inverse_aspect_ratio):

#     pose_data = pose_data.reshape(pose_data.shape[0], -1, 3)

#     mask = pose_data[:, :, 1] != OUT_OF_FRAME_NUM
#     pose_data[:, :, 1][mask] *= inverse_aspect_ratio  # scale y-axis values to recover original proportions

#     for frame_pose in pose_data:
        
#         if (np.any(frame_pose[L_SHOULDER_INDEX] == OUT_OF_FRAME_NUM) or 
#             np.any(frame_pose[R_SHOULDER_INDEX] == OUT_OF_FRAME_NUM)):
#             frame_pose[:] = OUT_OF_FRAME_NUM
#             continue

#         l_shoulder = frame_pose[L_SHOULDER_INDEX]
#         r_shoulder = frame_pose[R_SHOULDER_INDEX]
#         neck = (l_shoulder + r_shoulder) / 2
#         dist = np.linalg.norm(l_shoulder - r_shoulder)

#         mask = (frame_pose != OUT_OF_FRAME_NUM).all(axis=1)
#         frame_pose[mask] -= neck
#         frame_pose[mask] /= dist

#     # Flatten back to original shape
#     return pose_data.reshape(pose_data.shape[0], -1)


# # hands and face should be normalized before the main pose, because some keypoints that are changed here are needed for the others
# def normalize_main_pose(frame_pose):

#     main_pose = frame_pose[2*21 : 2*21+len(POSE_INDICES)]

#     l_shoulder = frame_pose[L_SHOULDER_INDEX]
#     r_shoulder = frame_pose[R_SHOULDER_INDEX]
#     l_wrist = frame_pose[0]
#     r_wrist = frame_pose[21]
#     nose = frame_pose[NOSE_INDEX]
    
#     if (l_shoulder[0] == OUT_OF_FRAME_NUM or 
#         r_shoulder[0] == OUT_OF_FRAME_NUM):
        
#         main_pose[:] = OUT_OF_FRAME_NUM
#         l_wrist[:] = OUT_OF_FRAME_NUM
#         r_wrist[:] = OUT_OF_FRAME_NUM
#         nose[:] = OUT_OF_FRAME_NUM # nose
#         return
    
#     neck = (l_shoulder + r_shoulder) / 2
#     dist = np.linalg.norm(l_shoulder - r_shoulder)

#     mask = (main_pose != OUT_OF_FRAME_NUM).all(axis=1)
#     main_pose[mask] -= neck
#     main_pose[mask] /= dist

#     if l_wrist[0] != OUT_OF_FRAME_NUM:
#         l_wrist -= neck
#         l_wrist /= dist

#     if r_wrist[0] != OUT_OF_FRAME_NUM:  
#         r_wrist -= neck
#         r_wrist /= dist

#     if nose[0] != OUT_OF_FRAME_NUM:
#         nose -= neck
#         nose /= dist
    

# def normalize_hands(frame_pose):

#     for offset in [0, 21]:

#         hand = frame_pose[offset:offset+21]

#         wrist = frame_pose[offset]
#         middle_mcp = frame_pose[offset+9]

#         wrist_copy = wrist.copy()

#         if (wrist[0] == OUT_OF_FRAME_NUM or 
#             middle_mcp[0] == OUT_OF_FRAME_NUM):

#             hand[:] = OUT_OF_FRAME_NUM
#             wrist[:] = wrist_copy
#             continue

#         dist = np.linalg.norm(wrist - middle_mcp)

#         mask = (hand != OUT_OF_FRAME_NUM).all(axis=1)
#         hand[mask] -= wrist
#         hand[mask] /= dist

#         wrist[:] = wrist_copy
    

# def normalize_face(frame_pose):

#     face = frame_pose[2*21 + len(POSE_INDICES) :]

#     l_ear = frame_pose[L_EAR_INDEX]
#     r_ear = frame_pose[R_EAR_INDEX]
#     nose = frame_pose[NOSE_INDEX]

#     nose_copy = nose.copy()

#     if (l_ear[0] == OUT_OF_FRAME_NUM or 
#         r_ear[0] == OUT_OF_FRAME_NUM or
#         nose[0] == OUT_OF_FRAME_NUM):
        
#         face[:] = OUT_OF_FRAME_NUM
#         nose[:] = nose_copy # Hau zertarako jarri nuen??
#         return
    
#     dist = np.linalg.norm(l_ear - r_ear)

#     mask = (face != OUT_OF_FRAME_NUM).all(axis=1)
#     face[mask] -= nose
#     face[mask] /= dist

#     nose[:] = nose_copy


# def normalize(pose_data, inverse_aspect_ratio):

#     pose_data = pose_data.reshape(pose_data.shape[0], -1, 3)

#     mask = pose_data[:, :, 1] != OUT_OF_FRAME_NUM
#     pose_data[:, :, 1][mask] *= inverse_aspect_ratio  # scale y-axis values to recover original proportions

#     for frame_pose in pose_data:

#         normalize_hands(frame_pose)
#         normalize_face(frame_pose)
#         normalize_main_pose(frame_pose)

#     return pose_data.reshape(pose_data.shape[0], -1)


# def normalize_hands_shoulder_dist(frame_pose, dist):

#     for offset in [0, 21]:
#         hand = frame_pose[offset:offset+21]

#         wrist = frame_pose[offset]
#         wrist_copy = wrist.copy()

#         if wrist[0] == OUT_OF_FRAME_NUM:
#             hand[:] = OUT_OF_FRAME_NUM
#             continue

#         mask = (hand != OUT_OF_FRAME_NUM).all(axis=1)
#         hand[mask] -= wrist
#         hand[mask] /= dist

#         wrist[:] = wrist_copy


# def normalize_face_shoulder_dist(frame_pose, dist):
    
#     face = frame_pose[2*21 + len(POSE_INDICES) :]

#     nose = frame_pose[NOSE_INDEX]
#     nose_copy = nose.copy()

#     if nose[0] == OUT_OF_FRAME_NUM:
#         face[:] = OUT_OF_FRAME_NUM
#         return

#     mask = (face != OUT_OF_FRAME_NUM).all(axis=1)
#     face[mask] -= nose
#     face[mask] /= dist

#     nose[:] = nose_copy


# def normalize_main_pose_shoulder_dist(frame_pose, dist, neck):
    
#     main_pose = frame_pose[2*21 : 2*21+len(POSE_INDICES)]

#     l_wrist = frame_pose[0]
#     r_wrist = frame_pose[21]
#     nose = frame_pose[NOSE_INDEX]

#     mask = (main_pose != OUT_OF_FRAME_NUM).all(axis=1)
#     main_pose[mask] -= neck
#     main_pose[mask] /= dist

#     if l_wrist[0] != OUT_OF_FRAME_NUM:
#         l_wrist -= neck
#         l_wrist /= dist

#     if r_wrist[0] != OUT_OF_FRAME_NUM:  
#         r_wrist -= neck
#         r_wrist /= dist

#     if nose[0] != OUT_OF_FRAME_NUM:
#         nose -= neck
#         nose /= dist


# def normalize_only_shoulder_dist(pose_data, inverse_aspect_ratio):

#     pose_data = pose_data.reshape(pose_data.shape[0], -1, 3)

#     mask = pose_data[:, :, 1] != OUT_OF_FRAME_NUM
#     pose_data[:, :, 1][mask] *= inverse_aspect_ratio  # scale y-axis values to recover original proportions

#     for frame_pose in pose_data:

#         if (frame_pose[L_SHOULDER_INDEX][0] == OUT_OF_FRAME_NUM or 
#             frame_pose[R_SHOULDER_INDEX][0] == OUT_OF_FRAME_NUM):
#             frame_pose[:] = OUT_OF_FRAME_NUM
#             continue

#         l_shoulder = frame_pose[L_SHOULDER_INDEX]
#         r_shoulder = frame_pose[R_SHOULDER_INDEX]
#         neck = (l_shoulder + r_shoulder) / 2
#         dist = np.linalg.norm(l_shoulder - r_shoulder)

#         normalize_hands_shoulder_dist(frame_pose, dist)
#         normalize_face_shoulder_dist(frame_pose, dist)
#         normalize_main_pose_shoulder_dist(frame_pose, dist, neck)

#     return pose_data.reshape(pose_data.shape[0], -1)


def impute(hand_array, interpolation_indices, partial_indices, max_gap=10):

    interpolated_indices = len(hand_array) * [False]
    
    for interpolation_dict in interpolation_indices:
        missing = interpolation_dict['missing_index']
        previous = interpolation_dict['previous_index']
        next_index = interpolation_dict.get('next_index', None)
        if (previous 
            and next_index 
            and not partial_indices[previous] 
            and not partial_indices[next_index]
        ):
            gap = next_index - previous
            if gap <= max_gap:
                previous_pose = hand_array[previous]
                next_pose = hand_array[next_index]
                interpolation_factor = (missing - previous) / gap
                interpolated = (1 - interpolation_factor) * previous_pose + interpolation_factor * next_pose
                hand_array[missing] = interpolated
                interpolated_indices[missing] = True

    return interpolated_indices


def temporal_imputation(pose_data):

    pose_data = pose_data.reshape(pose_data.shape[0], -1, 3)

    hand_offsets = [0, 21]

    for offset in hand_offsets:
        interpolation_indices = []
        next_indices = []
        partial_indices = len(pose_data) * [False]
        previous_index = None
        for i, pose in enumerate(pose_data):
            if np.all(pose[offset:offset+21] == -100):
                interpolation_indices.append({
                    'missing_index': i,
                    'previous_index': previous_index
                })
            else:
                previous_index = i
                if interpolation_indices and interpolation_indices[-1]['missing_index'] == i - 1:
                    next_indices.append(i)
                if np.any(pose[offset:offset+21] == -100):
                    partial_indices[i] = True
        
        if not next_indices:
            return
        
        next_indices_iter = iter(next_indices)
        next_index = next(next_indices_iter)
        for interpolation_dict in interpolation_indices:
            if interpolation_dict['missing_index'] > next_index:
                try:
                    next_index = next(next_indices_iter)
                except StopIteration:
                    break
            interpolation_dict['next_index'] = next_index
                
        hand_array = pose_data[:, offset:offset+21]
        impute(hand_array, interpolation_indices, partial_indices, max_gap=10)


def extract_interval_captions(caption_list, start_time, end_time, context_seconds=17):

    prev = []
    curr = []
    next_ = []

    prev_start = start_time - context_seconds * 1000
    next_end = end_time + context_seconds * 1000

    for caption in caption_list:
        if caption['start_time'] > prev_start:
            if caption['start_time'] < start_time:
                prev.append(caption)
            elif caption['end_time'] < end_time:
                curr.append(caption)
            elif caption['end_time'] < next_end:
                next_.append(caption)
            else:
                break

    return prev, curr, next_


def average_caption_duration(captions):
    total_duration = 0

    for caption in captions:
        duration = caption['end_time'] - caption['start_time']
        total_duration += duration

    return total_duration / len(captions) if len(captions) else 0


def convert_captions_dataset(captions_ds):
    
    captions = [{
        'sentence': row['sentence'].decode('utf-8'),
        'start_time': row['start_time'],
        'end_time': row['end_time']
    } for row in captions_ds[:]
    ]

    return captions


def filter_captions(captions, max_characters=200, max_seconds=20, min_seconds=0.5):

    filtered_captions = []
    for caption in captions:
        if len(caption['sentence']) > max_characters:
            continue
        if (caption['end_time'] - caption['start_time']) > max_seconds * 1000:
            continue
        if (caption['end_time'] - caption['start_time']) < min_seconds * 1000:
            continue
        filtered_captions.append(caption)

    return filtered_captions


def read_to_datasets(data_paths, frame_freq=2, chunk_seconds=34, max_characters=200, max_seconds=20, min_seconds=0.5, captions_langs=None):

    train_caption_real = []
    dev_caption_real = []

    train_caption_synthetic = []
    dev_caption_synthetic = []
    
    train_random_real = []
    dev_random_real = []
    
    train_random_synthetic = []
    dev_random_synthetic = []

    for data_path in data_paths:

        with h5py.File(data_path, 'r') as h5_file:

            for id, video_group in tqdm(h5_file.items()):
                
                sampled_fps = video_group.attrs['fps'] / frame_freq
                aprox_chunk_length = round(chunk_seconds * sampled_fps)
                keypoint_ds = video_group['processed_keypoints']
                
                if keypoint_ds.shape[0] == 0:
                    continue
                
                chunks_indices = list(chunked_even(range(keypoint_ds.shape[0]), aprox_chunk_length))

                #TODO: momentuz txapuza hau dev multzo bat izateko. Pentsatu datu gehiagorekin balioko duen zerbait
                #split = 'dev' if id[0] == '1' else 'train' # LSE
                # split = 'dev' if id == '14SjcMwwNhM' or id == '1y8vXjLQWL0' or id == '23tQHcy3VnI' or id == '14diAi-BS40' or id == '--7t6QjMwpY' or id == '--pq96V-6DA' or id == '-0VIwubCjpM' else 'train' # ASL # 14diAi-BS40 EZ DAGO!
                split = 'train'

                for lang, captions_ds in video_group['captions'].items():

                    if captions_langs and lang not in captions_langs:
                        continue

                    captions = convert_captions_dataset(captions_ds)
                    captions = filter_captions(captions, max_characters=max_characters, max_seconds=max_seconds, min_seconds=min_seconds)
                    avg_cap_dur = average_caption_duration(captions)

                    for caption in captions:

                        start_frame = round(caption['start_time'] / 1000 * sampled_fps)
                        end_frame = round(caption['end_time'] / 1000 * sampled_fps)

                        if start_frame > keypoint_ds.shape[0]:
                            continue

                        if not end_frame > start_frame:
                            continue

                        if end_frame > keypoint_ds.shape[0]:
                            end_frame = keypoint_ds.shape[0]

                        instance = {
                            'instance_type': 'caption_clip_slt',
                            'tgt_sentence': caption['sentence'],
                            'src_lang': video_group.attrs['sign_lang'],
                            'tgt_lang': lang,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'video_id': id,
                            'all_captions': None,
                            'avg_cap_dur': None,
                            'sampled_fps': None,
                        }

                        if split == 'train':
                            if captions_ds.attrs['is_synthetic']:
                                train_caption_synthetic.append(instance)
                            else:
                                train_caption_real.append(instance)
                        else:
                            if captions_ds.attrs['is_synthetic']:
                                dev_caption_synthetic.append(instance)
                            else:
                                dev_caption_real.append(instance)

                    previous_indices = chunks_indices[0]
                    for current_indices in chunks_indices[1:]:
                        
                        instance = {
                            'instance_type': 'random_clip_slt',
                            'all_captions': captions,
                            'avg_cap_dur': avg_cap_dur,
                            'sampled_fps': sampled_fps,
                            'start_frame': previous_indices[0],
                            'end_frame': current_indices[-1] + 1,
                            'src_lang': video_group.attrs['sign_lang'],
                            'tgt_lang': lang,
                            'video_id': id,
                            'tgt_sentence': None,
                        }

                        if split == 'train':
                            if captions_ds.attrs['is_synthetic']:
                                train_random_synthetic.append(instance)
                            else:
                                train_random_real.append(instance)
                        else:
                            if captions_ds.attrs['is_synthetic']:
                                dev_random_synthetic.append(instance)
                            else:
                                dev_random_real.append(instance)

                        previous_indices = current_indices

    # TODO: dev guztiak kendu
    return (Dataset.from_list(train_caption_real), 
            Dataset.from_list(dev_caption_real),
            Dataset.from_list(train_caption_synthetic),  
            Dataset.from_list(dev_caption_synthetic),
            Dataset.from_list(train_random_real),
            Dataset.from_list(dev_random_real),
            Dataset.from_list(train_random_synthetic),
            Dataset.from_list(dev_random_synthetic))


def sentence_level_hdf5_to_dataset(data_path, dataset_name, src_lang, tgt_lang):

    instances = []

    with h5py.File(data_path, 'r') as f:
        for sentence_id in f.keys():
            # These datasets won't be concatenated with others, so the fields in each instance
            # don't need to be compatible with other instance types.
            instances.append({
                'instance_type': 'sentence_level_dataset_slt',
                'src_lang': src_lang,
                'tgt_lang': tgt_lang,
                'dataset_name': dataset_name,
                'sentence_id': sentence_id
            })

    return Dataset.from_list(instances)


def convert_opus_dataset(instance):

    src_lang, tgt_lang = instance['translation'].keys()

    return {
        'instance_type': 'text_mt',
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
        'src_sentence': instance['translation'][src_lang],
        'tgt_sentence': instance['translation'][tgt_lang],
        'start_frame': None,
        'end_frame': None,
        'video_id': None,
        'all_captions': None,
        'avg_cap_dur': None,
        'sampled_fps': None
    }


def reverse_mt_dataset(instance):

    return {
        'src_lang': instance['tgt_lang'],
        'tgt_lang': instance['src_lang'],
        'src_sentence': instance['tgt_sentence'],
        'tgt_sentence': instance['src_sentence'],
    }


def load_mt_datasets(mt_pairs, max_characters=200):
    
    mt_datasets = []
    reverse_mt_datasets = []
    for pair in mt_pairs:
        opus_ds = load_dataset('Helsinki-NLP/opus-100', pair, split='train')
        mt_ds = opus_ds.map(
            convert_opus_dataset,
            remove_columns=opus_ds.column_names
        )
        mt_ds = mt_ds.filter(
            lambda x: len(x["src_sentence"]) <= max_characters
            and len(x["tgt_sentence"]) <= max_characters
        )
        reverse_mt_ds = mt_ds.map(reverse_mt_dataset)
        mt_datasets.append(mt_ds)
        reverse_mt_datasets.append(reverse_mt_ds)
    
    return concatenate_datasets(mt_datasets), concatenate_datasets(reverse_mt_datasets)


def get_text_with_timestamps(captions, interval_start_time=0):
    
    text_with_timestamps = []
    for caption in captions:
        start_time = round((caption['start_time'] - interval_start_time) / 1000, 1)
        end_time = round((caption['end_time'] - interval_start_time) / 1000, 1)
        text_with_timestamps.append(f"{start_time} {end_time} {caption['sentence']}")
    
    return '\n'.join(text_with_timestamps)


def get_left_edge(prev_captions, start_time):
    if not prev_captions:
        return 0
    return max(0, prev_captions[-1]['end_time'] - start_time)


def get_right_edge(next_captions, start_time, end_time):
    length = end_time - start_time
    if not next_captions:
        return length
    return min(length, next_captions[0]['start_time'] - start_time)


class DataCollator:
    def __init__(
        self, 
        data_paths, 
        tokenizer, 
        model_input_dim, 
        pose_pad_value=0.0, 
        chunk_seconds=34, 
        min_duration_seconds=17,
        max_tokens=800
    ):

        self.data_paths = data_paths
        self.files = None
        
        self.tokenizer = tokenizer
        self.pose_pad_value = pose_pad_value
        self.model_input_dim = model_input_dim
        self.chunk_seconds = chunk_seconds
        self.min_duration_seconds = min_duration_seconds
        self.max_tokens = max_tokens


    def _ensure_files_open(self):
        
        if self.files is None:
            self.files = {
                file_key: h5py.File(data_path, 'r')
                for file_key, data_path in self.data_paths.items()
            }
    

    def _caption_level_input_text(self, src_lang, tgt_lang):
        return f'slt 1 src:{src_lang} tgt:{tgt_lang}\n'
    
    
    def _process_sentence_level_dataset_slt_instance(self, instance):

        sentence_group = self.files[instance['dataset_name']][instance['sentence_id']]
        pose = torch.tensor(sentence_group['processed_keypoints'][:], dtype=torch.float32)
        input_text = self._caption_level_input_text(instance["src_lang"], instance["tgt_lang"])
        target_text = sentence_group.attrs['sentence']

        return input_text, pose, target_text

    
    def _process_caption_clip_slt_instance(self, instance):
        
        video_group = self.files[instance['src_lang']][instance['video_id']]
        start_frame = instance['start_frame']
        end_frame = instance['end_frame']
        pose = torch.tensor(video_group['processed_keypoints'][start_frame:end_frame], dtype=torch.float32)
        input_text = self._caption_level_input_text(instance["src_lang"], instance["tgt_lang"])
        target_text = instance['tgt_sentence']    

        return input_text, pose, target_text
    

    def _process_random_clip_slt_instance(self, instance):
        
        video_group = self.files[instance['src_lang']][instance['video_id']]
        start_frame = instance['start_frame']
        end_frame = instance['end_frame']
        chunk_pose = video_group['processed_keypoints'][start_frame:end_frame]

        n_frame_count = len(chunk_pose) // 2
        m_frame_count = int(self.min_duration_seconds * instance['sampled_fps'])
        start_frame = random.randint(0, n_frame_count)
        if random.random() < 0.8:
            end_frame = start_frame + n_frame_count
        else:
            end_frame = start_frame + random.randint(m_frame_count, n_frame_count)
        
        instance_start_frame = instance['start_frame']
        start_time = (instance_start_frame + start_frame) / instance['sampled_fps'] * 1000
        end_time = (instance_start_frame + end_frame) / instance['sampled_fps'] * 1000
        prev_captions, curr_captions, next_captions = extract_interval_captions(
            instance['all_captions'], 
            start_time, 
            end_time,
            context_seconds=self.min_duration_seconds
        )

        pose = torch.tensor(chunk_pose[start_frame:end_frame], dtype=torch.float32)

        input_text = 'slt'

        timed = False
        guess_lang_id = False

        if random.random() < 0.8:
            # timed
            input_text += ' timed'
            timed = True
            if random.random() < 0.5:
                # avg cap dur
                input_text += f' avg:{round(instance["avg_cap_dur"] / 1000, 1)}'
        else:
            # untimed
            input_text += ' untimed'
        # left edge
        left_edge = get_left_edge(prev_captions, start_time)
        input_text += f' start:{round(left_edge / 1000, 1)}'
        if random.random() < 0.9:
            # src/trg lang
            input_text += f' src:{instance["src_lang"]} tgt:{instance["tgt_lang"]}'
            if random.random() < 0.9:
                # untimed prev captions
                input_text += '\n' + '\n'.join(caption['sentence'] for caption in prev_captions)
                if random.random() < 0.2:
                    # right edge
                    right_edge = get_right_edge(next_captions, start_time, end_time)
                    input_text += f'\nend:{round(right_edge / 1000, 1)}'
                    # untimed next captions
                    input_text += '\n' + '\n'.join(caption['sentence'] for caption in next_captions)
        else:
            # lang id
            input_text += ' guess_lang_id'
            guess_lang_id = True

        input_text += '\n'

        if timed:
            target_text = get_text_with_timestamps(curr_captions, interval_start_time=start_time)
        else:
            target_text = '\n'.join(caption['sentence'] for caption in curr_captions) # separazioa egotea nahi dugu?
        
        if guess_lang_id:
            target_text = f'{instance["src_lang"]} {instance["tgt_lang"]}\n{target_text}'

        return input_text, pose, target_text


    def _process_text_mt_instance(self, instance):
        
        input_text = f'mt src:{instance["src_lang"]} tgt:{instance["tgt_lang"]}\n{instance["src_sentence"]}'

        return input_text, instance['tgt_sentence']


    def __call__(self, batch):

        self._ensure_files_open()

        input_texts = []
        poses = []
        target_texts = []
        
        for instance in batch:

            if instance['instance_type'] == 'sentence_level_dataset_slt':
                input_text, pose, target_text = self._process_sentence_level_dataset_slt_instance(instance)
            elif instance['instance_type'] == 'caption_clip_slt':
                input_text, pose, target_text = self._process_caption_clip_slt_instance(instance)
            elif instance['instance_type'] == 'random_clip_slt':
                input_text, pose, target_text = self._process_random_clip_slt_instance(instance)
            elif instance['instance_type'] == 'text_mt':
                input_text, target_text = self._process_text_mt_instance(instance)
                pose = torch.zeros((0, self.model_input_dim))
            else:
                raise ValueError(f"Unknown instance type: {instance['instance_type']}")

            assert pose.shape[1] == self.model_input_dim, (
                f'Processed keypoints dimension {pose.shape[1]} '
                f'does not match model input dimension {self.model_input_dim}'
            )
            
            input_texts.append(input_text)
            poses.append(pose)
            target_texts.append(target_text)

            # DEBUG
            '''
            print('video_id:', instance['video_id'])
            print()
            print('input_text:')
            print(input_text)
            print()
            print('target_text:')
            print(target_text)
            print()
            print('---')
            print()
            '''

        encoded = self.tokenizer(input_texts, padding=True, return_tensors='pt')
        input_ids = encoded['input_ids']
        text_attention_mask = encoded['attention_mask']
        
        # Shape: (batch_size, max_seq_len, model_input_dim)
        padded_poses = pad_sequence(poses, batch_first=True, padding_value=self.pose_pad_value)
        pose_attention_mask = (~torch.all(padded_poses == 0.0, dim=-1)).long()

        tokenized_target = self.tokenizer(target_texts, padding=True, return_tensors="pt")["input_ids"]
        tokenized_target[tokenized_target == self.tokenizer.pad_token_id] = -100

        batch_dict = {
            'input_ids': input_ids[:, :self.max_tokens],
            'input_vectors': padded_poses[:, :self.max_tokens],
            'text_attention_mask': text_attention_mask[:, :self.max_tokens],
            'vectors_attention_mask': pose_attention_mask[:, :self.max_tokens],
            'labels': tokenized_target[:, :self.max_tokens]
        }
        
        return batch_dict
        

class DebugTextOnlyDataCollator:
    def __init__(
        self, 
        data_paths, 
        tokenizer, 
        model_input_dim, 
        pose_pad_value=0.0, 
        chunk_seconds=34, 
        min_duration_seconds=17
    ):

        self.data_paths = data_paths
        self.files = None
        
        self.tokenizer = tokenizer
        self.pose_pad_value = pose_pad_value
        self.model_input_dim = model_input_dim
        self.chunk_seconds = chunk_seconds
        self.min_duration_seconds = min_duration_seconds


    # def _ensure_files_open(self):
        
    #     if self.files is None:
    #         self.files = {
    #             lang: h5py.File(data_path, 'r')
    #             for lang, data_path in self.data_paths.items()
    #         }


    def _process_caption_clip_slt_instance(self, instance):
        
        # video_group = self.files[instance['src_lang']][instance['video_id']]
        # start_frame = instance['start_frame']
        # end_frame = instance['end_frame']
        # pose = torch.tensor(video_group['processed_keypoints'][start_frame:end_frame], dtype=torch.float32)
        input_text = f'slt 1 src:{instance["src_lang"]} tgt:{instance["tgt_lang"]}\n'
        target_text = instance['tgt_sentence']    

        return input_text, None, target_text
    

    def _process_random_clip_slt_instance(self, instance):
        
        # video_group = self.files[instance['src_lang']][instance['video_id']]
        start_frame = instance['start_frame']
        end_frame = instance['end_frame']
        # chunk_pose = video_group['processed_keypoints'][start_frame:end_frame]

        n_frame_count = int(self.chunk_seconds * instance['sampled_fps'])  #len(chunk_pose) // 2
        m_frame_count = int(self.min_duration_seconds * instance['sampled_fps'])
        start_frame = random.randint(0, n_frame_count)
        if random.random() < 0.8:
            end_frame = start_frame + n_frame_count
        else:
            end_frame = start_frame + random.randint(m_frame_count, n_frame_count)
        
        instance_start_frame = instance['start_frame']
        start_time = (instance_start_frame + start_frame) / instance['sampled_fps'] * 1000
        end_time = (instance_start_frame + end_frame) / instance['sampled_fps'] * 1000
        prev_captions, curr_captions, next_captions = extract_interval_captions(
            instance['all_captions'], 
            start_time, 
            end_time,
            context_seconds=self.min_duration_seconds
        )

        # pose = torch.tensor(chunk_pose[start_frame:end_frame], dtype=torch.float32)

        input_text = 'slt'

        timed = False
        guess_lang_id = False

        if random.random() < 0.8:
            # timed
            input_text += ' timed'
            timed = True
            if random.random() < 0.5:
                # avg cap dur
                input_text += f' avg:{round(instance["avg_cap_dur"] / 1000, 1)}'
        else:
            # untimed
            input_text += ' untimed'
        # left edge
        left_edge = get_left_edge(prev_captions, start_time)
        input_text += f' start:{round(left_edge / 1000, 1)}'
        if random.random() < 0.9:
            # src/trg lang
            input_text += f' src:{instance["src_lang"]} tgt:{instance["tgt_lang"]}'
            if random.random() < 0.9:
                # untimed prev captions
                input_text += '\n' + '\n'.join(caption['sentence'] for caption in prev_captions)
                if random.random() < 0.2:
                    # right edge
                    right_edge = get_right_edge(next_captions, start_time, end_time)
                    input_text += f'\nend:{round(right_edge / 1000, 1)}'
                    # untimed next captions
                    input_text += '\n' + '\n'.join(caption['sentence'] for caption in next_captions)
        else:
            # lang id
            input_text += ' guess_lang_id'
            guess_lang_id = True

        input_text += '\n'

        if timed:
            target_text = get_text_with_timestamps(curr_captions, interval_start_time=start_time)
        else:
            target_text = '\n'.join(caption['sentence'] for caption in curr_captions) # separazioa egotea nahi dugu?
        
        if guess_lang_id:
            target_text = f'{instance["src_lang"]} {instance["tgt_lang"]}\n{target_text}'

        return input_text, None, target_text


    def _process_text_mt_instance(self, instance):
        
        input_text = f'mt src:{instance["src_lang"]} tgt:{instance["tgt_lang"]}\n{instance["src_sentence"]}'

        return input_text, instance['tgt_sentence']


    def __call__(self, batch):

        # self._ensure_files_open()

        input_texts = []
        # poses = []
        target_texts = []
        
        for instance in batch:

            if instance['instance_type'] == 'caption_clip_slt':
                input_text, pose, target_text = self._process_caption_clip_slt_instance(instance)
            elif instance['instance_type'] == 'random_clip_slt':
                input_text, pose, target_text = self._process_random_clip_slt_instance(instance)
            elif instance['instance_type'] == 'text_mt':
                input_text, target_text = self._process_text_mt_instance(instance)
                pose = torch.zeros((0, self.model_input_dim))
            else:
                raise ValueError(f"Unknown instance type: {instance['instance_type']}")

            # assert pose.shape[1] == self.model_input_dim, (
            #     f'Processed keypoints dimension {pose.shape[1]} '
            #     f'does not match model input dimension {self.model_input_dim}'
            # )
            
            input_texts.append(input_text)
            # poses.append(pose)
            target_texts.append(target_text)

            # DEBUG
            '''
            print('video_id:', instance['video_id'])
            print()
            print('input_text:')
            print(input_text)
            print()
            print('target_text:')
            print(target_text)
            print()
            print('---')
            print()
            '''

        encoded = self.tokenizer(input_texts, padding=True, return_tensors='pt')
        input_ids = encoded['input_ids']
        text_attention_mask = encoded['attention_mask']
        
        # Shape: (batch_size, max_seq_len, model_input_dim)
        # padded_poses = pad_sequence(poses, batch_first=True, padding_value=self.pose_pad_value)
        # pose_attention_mask = (~torch.all(padded_poses == 0.0, dim=-1)).long()

        tokenized_target = self.tokenizer(target_texts, padding=True, return_tensors="pt")["input_ids"]
        tokenized_target[tokenized_target == self.tokenizer.pad_token_id] = -100

        batch_dict = {
            'input_ids': input_ids,
            # 'input_vectors': padded_poses,
            # 'text_attention_mask': text_attention_mask,
            # 'vectors_attention_mask': pose_attention_mask,
            'labels': tokenized_target,
            'input_texts': input_texts,
            'target_texts': target_texts,
            'video_ids': [instance['video_id'] for instance in batch],
            'instance_types': [instance['instance_type'] for instance in batch]
        }
        
        return batch_dict
