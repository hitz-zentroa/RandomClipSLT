"""Add a custom protocol from CSV columns sentence_id,split (without extraction)."""
import argparse
import csv
import hashlib
from pathlib import Path
import h5py
import yaml
from data.sentence_datasets import DATASETS, dataset_path, initialize_store, write_partitions


def main(args, config):
    if not args.protocol or '/' in args.protocol or args.protocol in ('.', '..'):
        raise ValueError('Invalid protocol name.')
    source = Path(args.manifest)
    with source.open(newline='', encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    ids = set()
    with h5py.File(dataset_path(config, args.dataset), 'r+') as file:
        initialize_store(file, args.dataset)
        if args.protocol in file['splits']:
            raise ValueError('Protocol already exists; choose a new name to preserve reproducibility.')
        records = []
        for row in rows:
            sid, split = row['sentence_id'], row['split']
            if not split or '/' in split or split in ('.', '..') or sid in ids:
                raise ValueError('Invalid partition or repeated sentence ID.')
            if sid not in file['sentences'] or not file['sentences'][sid].attrs.get('complete', False):
                raise ValueError(f'{sid}: missing/incomplete keypoints.')
            ids.add(sid)
            records.append({'id': sid, 'memberships': {args.protocol: split}})
        if not records:
            raise ValueError('Empty manifest.')
        write_partitions(file, records, {'manifest_path': str(source),
                         'manifest_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--config-path', default='configs/config.yaml')
    args = parser.parse_args()
    with open(args.config_path) as file:
        main(args, yaml.safe_load(file))
