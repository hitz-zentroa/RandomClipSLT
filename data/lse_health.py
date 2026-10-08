"""Load the supplementary LSE-Health SD&TD workbook."""

import hashlib
from pathlib import Path

import pandas as pd


TEXT_COLUMN = 'PlainText: [S: What it was signed'
SPLIT_COLUMN = 'Split SD&TD'


def load_segments(args, config):
    workbook = Path(args.excel_path)
    splits = pd.read_excel(workbook, sheet_name='#Segments', skiprows=[1],
                           skipfooter=6, usecols=[1, 2, 3], engine='openpyxl')
    if splits.columns[0] != 'File' or splits.columns[2] != SPLIT_COLUMN:
        raise ValueError('Unexpected #Segments layout: expected File, count, Split SD&TD.')
    count_column = splits.columns[1]
    if splits['File'].isna().any() or splits['File'].duplicated().any():
        raise ValueError('Missing or duplicate video IDs in #Segments.')
    if not splits[SPLIT_COLUMN].isin(['Train', 'Dev', 'Test']).all():
        raise ValueError('Missing/invalid cached split values in #Segments.')
    split_by_video = splits.set_index('File')[SPLIT_COLUMN]
    # There is no footer here: the final six rows are real training segments.
    content = pd.read_excel(workbook, sheet_name='SegmentsContent',
                            usecols=['File', 'Start(ms)', 'End(ms)', TEXT_COLUMN],
                            engine='openpyxl')
    actual_counts = content.groupby('File').size()
    for _, row in splits.iterrows():
        if actual_counts.get(row['File'], 0) != row[count_column]:
            raise ValueError(f'{row["File"]}: segment count differs from #Segments.')
    selected = {'train': 'Train', 'val': 'Dev', 'test': 'Test'}[args.split]
    records = []
    counts = {'Train': 0, 'Dev': 0, 'Test': 0}
    for index, row in content.iterrows():
        video_id = row['File']
        if video_id not in split_by_video.index:
            raise ValueError(f'Excel row {index + 2}: unknown video {video_id!r}.')
        split = split_by_video.loc[video_id]
        counts[split] += 1
        text = row[TEXT_COLUMN]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Excel row {index + 2}: missing reference.')
        if split != selected:
            continue
        records.append({
            'id': f'{video_id}_row{index + 2:06d}',
            'video_path': str(Path(args.video_dir) / f'{video_id}.mp4'),
            'start_time': float(row['Start(ms)']) / 1000,
            'end_time': float(row['End(ms)']) / 1000,
            'attrs': {'sentence': text.lower().replace('¬', ''),
                      'video_id': video_id, 'source_row': index + 2},
        })
    print('SD&TD segment counts:', counts)
    return records, {
        'workbook_path': str(workbook),
        'workbook_sha256': hashlib.sha256(workbook.read_bytes()).hexdigest(),
        'split_protocol': 'SD&TD',
        'reference_transform': "lower().replace('¬', '')",
    }
