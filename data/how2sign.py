"""How2Sign sentence metadata (times in seconds)."""

from pathlib import Path

import pandas as pd


def load_segments(args, config):
    root = Path(config['data']['how2sign']['path'])
    csv_path = Path(args.csv_path) if args.csv_path else root / f'how2sign_realigned_{args.split}.csv'
    video_dir = Path(args.video_dir) if args.video_dir else root / f'{args.split}_raw_videos'
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
