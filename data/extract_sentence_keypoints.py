"""Parallel sentence extraction: python -m data.extract_sentence_keypoints --help.

Workers process one source video each; only the parent writes HDF5.
Timed clips reset the MediaPipe tracker per sentence, as in How2Sign.
"""

import argparse
from collections import defaultdict
from functools import partial
import math
import json
import warnings
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
from data.keypoint_schema import NUM_LANDMARKS
from data.sentence_datasets import (DATASETS, load_segments, dataset_path,
                                    initialize_store, write_partitions)


def process_video(task, model_complexity, dataset, metadata):
    video_path, segments = task
    if dataset == 'flores_lse':
        from data.flores_lse import resolve_video
        sources = json.loads(metadata['sources_json'])
        video_path = resolve_video(video_path, sources[segments[0]['memberships']['default']])
    probe = ffmpegio.probe.video_streams_basic(video_path)[0]
    cap = cv2.VideoCapture(video_path)
    results = []
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video {video_path}.')
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if frame_count > 0 and fps > 0 else None
        for segment in segments:
            start, end = segment['start_time'], segment['end_time']
            outside_video = start is not None and duration is not None and start >= duration
            if start is not None and not outside_video and not cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000):
                raise ValueError(f'{segment["id"]}: cannot seek to {start}.')
            frames = []
            with mp.solutions.holistic.Holistic(
                static_image_mode=False, model_complexity=model_complexity,
                refine_face_landmarks=True,
            ) as holistic:
                while not outside_video and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Preserve the existing How2Sign boundary convention.
                    if end is not None and cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 > end:
                        break
                    frames.append(process_frame(frame, holistic))
            if not frames:
                if start is None:
                    raise ValueError(f'{segment["id"]}: no decoded frames in the source video.')
                warnings.warn(
                    f'{segment["id"]}: no frames in annotated interval {start}–{end}; '
                    'saving empty keypoints and retaining the sentence.',
                    UserWarning,
                )
            keypoints = (np.stack(frames) if frames else
                         np.empty((0, NUM_LANDMARKS, 3), dtype=np.float32))
            results.append((segment, keypoints))
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
        for key, value in record['attrs'].items():
            if key == 'sentence' or key.startswith('text_'):
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f'{sentence_id}: missing {key}.')
        record['attrs']['order'] = order
        videos[record['video_path']].append(record)
    complexity = config['data']['keypoints']['mediapipe_model_complexity']
    worker = partial(process_video, model_complexity=complexity,
                     dataset=args.dataset, metadata=metadata)
    output = Path(dataset_path(config, args.dataset, args.h5_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not output.exists()
    print(f'Found {len(records)} sentences from {len(videos)} videos.', flush=True)
    # Spawn prevents workers from inheriting the parent's open HDF5 handle.
    with h5py.File(output, 'a') as file:
        initialize_store(file, args.dataset)
        if new_file:
            file.attrs['mediapipe_model_complexity'] = complexity
            file.attrs['mediapipe_version'] = mp.__version__
        recorded_complexity = file.attrs.get('mediapipe_model_complexity')
        if recorded_complexity is not None and recorded_complexity != complexity:
            raise ValueError(
                f'{output}: MediaPipe model complexity is {recorded_complexity}, '
                f'but the current config requests {complexity}.'
            )
        sentences = file['sentences']
        pending = defaultdict(list)
        for record in records:
            sid = record['id']
            if sid in sentences and sentences[sid].attrs.get('complete', False):
                group = sentences[sid]
                for key, value in record['attrs'].items():
                    if key == 'source_provenance_json' and key not in group.attrs:
                        # Legacy FLORES files did not record a resolved HF revision.
                        continue
                    if key != 'order' and group.attrs.get(key) != value:
                        raise ValueError(f'{sid}: metadata mismatch for {key}.')
                for key in ('start_time', 'end_time'):
                    if record[key] is not None and group.attrs.get(key) != record[key]:
                        raise ValueError(f'{sid}: interval mismatch.')
                continue
            if args.dataset != 'flores_lse' and not Path(record['video_path']).is_file():
                setting = (f'data.datasets.how2sign.sources.{record["memberships"]["default"]}.video_dir'
                           if args.dataset == 'how2sign' else 'data.datasets.lse_health.video_dir')
                raise FileNotFoundError(
                    f'Missing video: {record["video_path"]}. Follow the documented layout '
                    f'or override {setting} / --video-dir.'
                )
            if sid in sentences:
                del sentences[sid]
            pending[record['video_path']].append(record)
        remaining_sentences = sum(len(segments) for segments in pending.values())
        saved_sentences = len(records) - remaining_sentences
        print(
            f'Already saved: {saved_sentences} sentences. '
            f'Remaining: {remaining_sentences} sentences from {len(pending)} videos. '
            f'Videos skipped entirely: {len(videos) - len(pending)}.',
            flush=True,
        )
        recorded_version = file.attrs.get('mediapipe_version')
        if pending and recorded_version is not None and recorded_version != mp.__version__:
            warnings.warn(
                f'{output}: recorded MediaPipe version is {recorded_version}; '
                f'adding keypoints with {mp.__version__}. The recorded version is unchanged.',
                UserWarning,
            )
        # Lists describe the requested dataset even if an extraction is interrupted.
        # Readers check completeness of every selected sentence.
        write_partitions(file, records, metadata)
        with get_context('spawn').Pool(args.num_processes) as pool:
            for probe, results in tqdm(pool.imap_unordered(worker, pending.items()), total=len(pending)):
                for record, keypoints in results:
                    group = sentences.create_group(record['id'])
                    group.create_dataset('keypoints', data=keypoints, compression='gzip', compression_opts=6)
                    group.attrs.update({**record['attrs'], 'fps': float(probe['frame_rate']),
                                        'width': probe['width'], 'height': probe['height']})
                    if record['start_time'] is not None:
                        group.attrs['start_time'] = record['start_time']
                        group.attrs['end_time'] = record['end_time']
                    group.attrs['complete'] = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--split', default='all')
    parser.add_argument('--split-protocol')
    parser.add_argument('--h5-path', help='Override the configured output path.')
    parser.add_argument('--config-path', default='configs/config.yaml')
    parser.add_argument('--num-processes', type=int, default=1)
    parser.add_argument('--excel-path', help='LSE-Health supplementary splits workbook.')
    parser.add_argument('--csv-path', help='Optional How2Sign realigned CSV override.')
    parser.add_argument('--video-dir', help='Override the configured local video directory.')
    args = parser.parse_args()
    if args.num_processes < 1:
        parser.error('--num-processes must be positive')
    with open(args.config_path) as file:
        main(args, yaml.safe_load(file))
