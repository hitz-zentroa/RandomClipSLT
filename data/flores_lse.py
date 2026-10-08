"""FLORES-LSE metadata and revision-pinned video downloads."""

import csv
from pathlib import PurePosixPath


REPO_ID = 'HiTZ/flores-lse'
LANGUAGES = ('en', 'es', 'ca', 'eu', 'gl')


def load_segments(args, config):
    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().dataset_info(REPO_ID).sha
    metadata = hf_hub_download(repo_id=REPO_ID, repo_type='dataset',
                               filename=f'{args.split}/metadata.csv', revision=revision)
    with open(metadata, newline='', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        required = {'file_name', 'flores_id', *(f'text_{lang}' for lang in LANGUAGES)}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing FLORES-LSE metadata columns.')
        rows = list(reader)
    records = []
    for row in rows:
        if any(not row[column] for column in required):
            raise ValueError(f'Incomplete FLORES-LSE metadata: {row!r}')
        records.append({
            'id': PurePosixPath(row['file_name']).stem,
            'video_path': f'{args.split}/{row["file_name"]}',
            'start_time': None, 'end_time': None,
            'attrs': {'file_name': row['file_name'], 'flores_id': int(row['flores_id']),
                      **{f'text_{lang}': row[f'text_{lang}'] for lang in LANGUAGES}},
        })
    return records, {'repo_id': REPO_ID, 'revision': revision}


def resolve_video(filename, metadata):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=metadata['repo_id'], repo_type='dataset',
                           filename=filename, revision=metadata['revision'])
