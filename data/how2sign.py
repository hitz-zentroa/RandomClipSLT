"""How2Sign sentence metadata (times in seconds)."""

import pandas as pd
from data.dataset_paths import resolve_path


def load_segments(args, config):
    csv_path = resolve_path(
        config, 'how2sign', 'annotations_path', f'how2sign_realigned_{args.split}.csv',
        args.csv_path, split=args.split, must_exist=True,
    )
    video_dir = resolve_path(
        config, 'how2sign', 'video_dir', f'{args.split}_raw_videos',
        args.video_dir, split=args.split,
    )
    frame = pd.read_csv(csv_path, sep='\t')
    records = []
    for index, row in frame.iterrows():
        records.append({
            'id': row['SENTENCE_NAME'],
            'video_path': str(video_dir / f'{row["VIDEO_NAME"]}.mp4'),
            'start_time': float(row['START_REALIGNED']),
            'end_time': float(row['END_REALIGNED']),
            'attrs': {'sentence': row['SENTENCE'], 'video_id': row['VIDEO_NAME'],
                      'source_row': index + 2},
        })
    return records, {'metadata_path': str(csv_path)}
