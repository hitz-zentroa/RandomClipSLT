"""FLORES-LSE metadata and revision-pinned video downloads."""

import csv
from pathlib import PurePosixPath


REPO_ID = 'HiTZ/flores-lse'
LANGUAGES = ('en', 'es', 'ca', 'eu', 'gl')
NATIVE_SPLITS = ('dev', 'devtest')
PROTOCOLS = {'zs': ('val', 'test'), 'si': ('train', 'val', 'test'),
             'sd': ('train', 'val', 'test')}


def load_segments(args, config):
    from huggingface_hub import HfApi, hf_hub_download

    protocol = getattr(args, 'split_protocol', None) or 'zs'
    if protocol not in PROTOCOLS or args.split not in ('all', *PROTOCOLS[protocol]):
        raise ValueError('Invalid FLORES-LSE protocol or partition (use train, val, test).')
    settings = config['data']['datasets']['flores_lse']
    repo_id = settings.get('repo_id', REPO_ID)
    revision = HfApi().dataset_info(repo_id).sha
    provenance = {'repo_id': repo_id, 'revision': revision}
    labels = {'train': 'train', 'validation': 'val', 'test': 'test'}
    records, seen = [], set()
    for part in NATIVE_SPLITS:
        metadata = hf_hub_download(repo_id=repo_id, repo_type='dataset',
                                   filename=f'{part}/metadata.csv', revision=revision)
        with open(metadata, newline='', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            required = {'id', 'file_name', 'flores_id',
                        *(f'text_{lang}' for lang in LANGUAGES),
                        *(f'{p}_split' for p in PROTOCOLS)}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError('Missing FLORES-LSE metadata columns; the Hub dataset must include protocol assignments.')
            for row in reader:
                missing = sorted(column for column in required
                                 if not row[column] or not row[column].strip())
                if missing:
                    raise ValueError(f'Incomplete FLORES-LSE metadata ({", ".join(missing)}): {row!r}')
                sid = row['id']
                if sid in seen or sid != PurePosixPath(row['file_name']).stem:
                    raise ValueError(f'Invalid/duplicate FLORES-LSE recording ID: {sid}.')
                seen.add(sid)
                memberships = {}
                for p, partitions in PROTOCOLS.items():
                    assignment = labels.get(row[f'{p}_split'])
                    if assignment not in partitions:
                        raise ValueError(f'{sid}: invalid {p} assignment.')
                    memberships[p] = assignment
                # As for LSE-Health, partial extraction publishes only the requested partition.
                if args.split != 'all':
                    if memberships[protocol] != args.split:
                        continue
                    memberships = {protocol: args.split}
                records.append({
                    'id': sid, 'video_path': f'{part}/{row["file_name"]}',
                    'start_time': None, 'end_time': None, 'memberships': memberships,
                    'attrs': {'file_name': row['file_name'], 'flores_id': int(row['flores_id']),
                              **{f'text_{lang}': row[f'text_{lang}'] for lang in LANGUAGES}},
                })
    return records, provenance


def resolve_video(filename, metadata):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=metadata['repo_id'], repo_type='dataset',
                           filename=filename, revision=metadata['revision'])
