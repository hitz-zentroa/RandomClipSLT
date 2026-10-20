"""LSE-Health workbook references and verbatim video-level split assignments."""
import hashlib
import pandas as pd
from data.dataset_paths import resolve_path

TEXT_COLUMN = 'PlainText: [S: What it was signed'
PROTOCOLS = {'sd_td': 'Split SD&TD', 'sd_ti': 'Split SD&TI', 'si_ti': 'Split SI&TI'}


def load_segments(args, config):
    workbook = resolve_path(config, 'lse_health', 'annotations_path',
                            'LSE-Health-UVigo-Splits.xlsx', args.excel_path, must_exist=True)
    video_dir = resolve_path(config, 'lse_health', 'video_dir',
                             'Videos-LSE-Health-UVigo', args.video_dir)
    splits = pd.read_excel(workbook, sheet_name='#Segments', skiprows=[1],
                           skipfooter=6, usecols=[1, 2, 3, 4, 5], engine='openpyxl')
    if splits.columns[0] != 'File' or not set(PROTOCOLS.values()).issubset(splits.columns):
        raise ValueError('Unexpected #Segments columns.')
    if splits['File'].isna().any() or splits['File'].duplicated().any():
        raise ValueError('Missing/duplicate video IDs.')
    for column in PROTOCOLS.values():
        if not splits[column].isin(['Train', 'Dev', 'Test']).all():
            raise ValueError(f'Missing/invalid cached assignments in {column}.')
    count_column = splits.columns[1]
    indexed = splits.set_index('File')
    content = pd.read_excel(workbook, sheet_name='SegmentsContent',
                            usecols=['File', 'Start(ms)', 'End(ms)', TEXT_COLUMN], engine='openpyxl')
    counts = content.groupby('File').size()
    for _, row in splits.iterrows():
        if counts.get(row['File'], 0) != row[count_column]:
            raise ValueError(f'{row["File"]}: segment count mismatch.')
    protocol = getattr(args, 'split_protocol', None) or 'sd_td'
    if protocol not in PROTOCOLS or args.split not in ('all', 'train', 'val', 'test'):
        raise ValueError('Invalid LSE-Health protocol or partition.')
    labels = {'Train': 'train', 'Dev': 'val', 'Test': 'test'}
    records = []
    for index, row in content.iterrows():
        video_id, text = row['File'], row[TEXT_COLUMN]
        if video_id not in indexed.index:
            raise ValueError(f'Unknown video at Excel row {index + 2}.')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Missing reference at Excel row {index + 2}.')
        memberships = {p: labels[indexed.loc[video_id, c]] for p, c in PROTOCOLS.items()}
        # All protocols are populated only when the whole workbook is requested.
        if args.split != 'all':
            if memberships[protocol] != args.split:
                continue
            memberships = {protocol: args.split}
        records.append({'id': f'{video_id}_row{index + 2:06d}',
                        'video_path': str(video_dir / f'{video_id}.mp4'),
                        'start_time': float(row['Start(ms)']) / 1000,
                        'end_time': float(row['End(ms)']) / 1000,
                        'memberships': memberships,
                        'attrs': {'sentence': text.lower().replace('¬', ''),
                                  'video_id': video_id, 'source_row': index + 2}})
    return records, {'workbook_path': str(workbook),
                     'workbook_sha256': hashlib.sha256(workbook.read_bytes()).hexdigest(),
                     'reference_transform': "lower().replace('¬', '')",
                     'assignment_policy': 'Workbook video assignments, without manual corrections'}
