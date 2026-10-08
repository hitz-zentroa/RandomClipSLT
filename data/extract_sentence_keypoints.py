"""Parallel sentence extraction: python -m data.extract_sentence_keypoints --help.

Workers process one source video each; only the parent writes HDF5.
Timed clips reset the MediaPipe tracker per sentence, as in How2Sign.
"""

import argparse
from collections import defaultdict
from functools import partial
import math
from multiprocessing import get_context
from pathlib import Path

import cv2
import ffmpegio
import h5py
import mediapipe as mp
import numpy as np
from tqdm import tqdm
import yaml

from data.extract_keypoints import process_frame
from data.sentence_datasets import DATASETS, load_segments


def process_video(task, model_complexity, dataset, metadata):
    video_path, segments = task
    if dataset == 'flores_lse':
        from data.flores_lse import resolve_video
        video_path = resolve_video(video_path, metadata)
    probe = ffmpegio.probe.video_streams_basic(video_path)[0]
    cap = cv2.VideoCapture(video_path)
    results = []
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video {video_path}.')
        for segment in segments:
            start, end = segment['start_time'], segment['end_time']
            if start is not None and not cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000):
                raise ValueError(f'{segment["id"]}: cannot seek to {start}.')
            frames = []
            with mp.solutions.holistic.Holistic(
                static_image_mode=False, model_complexity=model_complexity,
                refine_face_landmarks=True,
            ) as holistic:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Preserve the existing How2Sign boundary convention.
                    if end is not None and cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 > end:
                        break
                    frames.append(process_frame(frame, holistic))
            if not frames:
                raise ValueError(f'{segment["id"]}: no decoded frames.')
            results.append((segment, np.stack(frames)))
    finally:
        cap.release()
    return probe, results


def main(args, config):
    records, metadata = load_segments(args.dataset, args, config)
    if not records:
        raise ValueError('No segments selected.')
    videos = defaultdict(list)
    ids = set()
    for order, record in enumerate(records):
        sentence_id = record['id']
        if not isinstance(sentence_id, str) or not sentence_id or '/' in sentence_id or sentence_id in ids:
            raise ValueError(f'Invalid or duplicate sentence ID: {sentence_id!r}')
        ids.add(sentence_id)
        start, end = record['start_time'], record['end_time']
        if (start is None) != (end is None):
            raise ValueError(f'{sentence_id}: incomplete interval.')
        if start is not None and not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise ValueError(f'{sentence_id}: invalid interval {start}, {end}.')
        if args.dataset != 'flores_lse' and not Path(record['video_path']).is_file():
            raise FileNotFoundError(record['video_path'])
        for key, value in record['attrs'].items():
            if key == 'sentence' or key.startswith('text_'):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f'{sentence_id}: missing {key}.')
        record['attrs']['order'] = order
        videos[record['video_path']].append(record)
    complexity = config['data']['keypoints']['mediapipe_model_complexity']
    worker = partial(process_video, model_complexity=complexity,
                     dataset=args.dataset, metadata=metadata)
    output = Path(args.h5_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f'Extracting {len(records)} sentences from {len(videos)} videos.')
    # Spawn prevents workers from inheriting the parent's open HDF5 handle.
    with h5py.File(output, 'x') as file:
        file.attrs.update({**metadata, 'schema_version': 1, 'dataset': args.dataset,
                           'split': args.split, 'complete': False,
                           'expected_segments': len(records),
                           'mediapipe_model_complexity': complexity,
                           'mediapipe_version': mp.__version__,
                           'tracker_scope': 'sentence'})
        with get_context('spawn').Pool(args.num_processes) as pool:
            for probe, results in tqdm(pool.imap_unordered(worker, videos.items()), total=len(videos)):
                for record, keypoints in results:
                    group = file.create_group(record['id'])
                    group.create_dataset('keypoints', data=keypoints, compression='gzip', compression_opts=6)
                    group.attrs.update({**record['attrs'], 'fps': float(probe['frame_rate']),
                                        'width': probe['width'], 'height': probe['height'],
                                        'split': args.split})
                    if record['start_time'] is not None:
                        group.attrs['start_time'] = record['start_time']
                        group.attrs['end_time'] = record['end_time']
        if len(file) != len(records):
            raise ValueError('Incomplete extraction.')
        file.attrs['complete'] = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--split', required=True)
    parser.add_argument('--h5-path', required=True)
    parser.add_argument('--config-path', default='configs/config.yaml')
    parser.add_argument('--num-processes', type=int, default=1)
    parser.add_argument('--excel-path', help='LSE-Health supplementary splits workbook.')
    parser.add_argument('--csv-path', help='Optional How2Sign realigned CSV override.')
    parser.add_argument('--video-dir', help='Local videos; required for LSE-Health.')
    args = parser.parse_args()
    if args.num_processes < 1:
        parser.error('--num-processes must be positive')
    if args.dataset == 'lse_health' and (not args.excel_path or not args.video_dir):
        parser.error('LSE-Health requires --excel-path and --video-dir')
    with open(args.config_path) as file:
        main(args, yaml.safe_load(file))
