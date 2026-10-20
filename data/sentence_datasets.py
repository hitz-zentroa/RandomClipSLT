"""Sentence dataset selection and consolidated HDF5 storage (schema 2)."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
from data.dataset_paths import resolve_path

DATASETS = {
    'how2sign': {'splits': ('train', 'val', 'test'), 'source': 'ase', 'targets': ('en',)},
    'lse_health': {'splits': ('train', 'val', 'test'), 'source': 'ssp', 'targets': ('es',)},
    'flores_lse': {'splits': ('dev', 'devtest'), 'source': 'ssp', 'targets': ('en', 'es', 'ca', 'eu', 'gl')},
}


def protocol_name(name, protocol=None):
    return protocol or ('sd_td' if name == 'lse_health' else 'default')


def dataset_options(name, split, target=None):
    options = DATASETS[name]
    target = target or options['targets'][0]
    if target not in options['targets']:
        raise ValueError(f'{name}: target must be one of {options["targets"]}.')
    return options['source'], target


def dataset_path(config, name, override=None):
    return str(resolve_path(config, name, 'hdf5_path', 'hdf5/keypoints.h5', override))


def selection_key(name, protocol, split):
    return f'{name}:{protocol}:{split}'


def load_segments(name, args, config):
    """Return metadata for a native partition, or all native partitions."""
    module = importlib.import_module(f'data.{name}')
    if name == 'lse_health':
        return module.load_segments(args, config)
    if getattr(args, 'split_protocol', None) not in (None, 'default'):
        raise ValueError('Extract native partitions first; add custom protocols with data.set_sentence_splits.')
    parts = DATASETS[name]['splits'] if args.split == 'all' else (args.split,)
    records, provenance = [], {}
    resolved_revision = None
    for part in parts:
        if part not in DATASETS[name]['splits']:
            raise ValueError(f'Unknown native partition {part}.')
        child = SimpleNamespace(**{**vars(args), 'split': part})
        child.resolved_revision = resolved_revision
        rows, metadata = module.load_segments(child, config)
        resolved_revision = metadata.get('revision', resolved_revision)
        for row in rows:
            row['memberships'] = {'default': part}
            if name == 'flores_lse':
                row['attrs']['source_provenance_json'] = json.dumps(metadata, sort_keys=True)
        records.extend(rows)
        provenance[part] = metadata
    return records, {'sources_json': json.dumps(provenance, sort_keys=True)}


def initialize_store(file, name):
    if not file.attrs.get('migration_complete', True):
        raise ValueError('Interrupted migration: create a new migration output file.')
    if 'schema_version' in file.attrs:
        if file.attrs['schema_version'] != 2 or file.attrs['dataset'] != name:
            raise ValueError('Dataset/schema mismatch; migrate legacy files first.')
    elif len(file):
        raise ValueError('Legacy HDF5: migrate to a new file first.')
    file.attrs.update({'schema_version': 2, 'dataset': name})
    file.require_group('sentences')
    file.require_group('splits')


def write_partitions(file, records, provenance):
    """Publish ordered membership lists. Missing keypoints are allowed until extraction finishes."""
    partitions = {}
    for record in records:
        for protocol, split in record['memberships'].items():
            partitions.setdefault((protocol, split), []).append(record['id'])
    for (protocol, split), ids in partitions.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate IDs in {protocol}/{split}.')
        group = file['splits'].require_group(protocol)
        if split in group:
            del group[split]
        ds = group.create_dataset(split, data=ids, dtype=h5py.string_dtype('utf-8'))
        ds.attrs['provenance_json'] = json.dumps(provenance, sort_keys=True)


def load_sentence_dataset(path, name, split, target=None, protocol=None):
    from datasets import Dataset
    if not Path(path).is_file():
        raise FileNotFoundError(
            f'Missing consolidated dataset: {path}. Extract or migrate keypoints first, '
            f'or override data.datasets.{name}.hdf5_path.'
        )
    source, target = dataset_options(name, split, target)
    protocol = protocol_name(name, protocol)
    key = selection_key(name, protocol, split)
    instances, manifest = [], []
    with h5py.File(path, 'r') as file:
        if file.attrs.get('schema_version') != 2 or file.attrs.get('dataset') != name:
            raise ValueError('Expected consolidated HDF5 schema 2; migrate the legacy file first.')
        if not file.attrs.get('migration_complete', True):
            raise ValueError('Migration did not complete.')
        partition_path = f'splits/{protocol}/{split}'
        if partition_path not in file:
            raise ValueError(f'Partition {protocol}/{split} is not defined in {path}.')
        partition = file[partition_path]
        ids = partition.asstr()[:].tolist()
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate IDs in partition.')
        metadata = {'dataset': name, 'protocol': protocol, 'split': split,
                    'provenance_json': partition.attrs.get('provenance_json', '{}')}
        for sentence_id in ids:
            path_in_file = f'sentences/{sentence_id}'
            if path_in_file not in file or not file[path_in_file].attrs.get('complete', False):
                raise ValueError(f'{sentence_id}: missing/incomplete extraction.')
            group = file[path_in_file]
            # A completed extraction can intentionally contain zero frames for
            # an annotated interval outside its video. Keep its reference and
            # partition position so evaluation membership is unchanged.
            if 'keypoints' not in group or not isinstance(group['keypoints'], h5py.Dataset):
                raise ValueError(f'{sentence_id}: missing keypoints dataset.')
            reference_key = f'text_{target}' if name == 'flores_lse' else 'sentence'
            instances.append({'instance_type': 'sentence_level_dataset_slt',
                              'src_lang': source, 'tgt_lang': target, 'dataset_name': key,
                              'tgt_sentence': group.attrs[reference_key],
                              'height': group.attrs['height'], 'width': group.attrs['width'],
                              'sentence_id': path_in_file})
            manifest.append({'sentence_id': sentence_id, **{
                k: v.item() if hasattr(v, 'item') else v for k, v in group.attrs.items()}})
    if not instances:
        raise ValueError('Empty partition.')
    return Dataset.from_list(instances), manifest, metadata


def load_pretraining_validation(config):
    selection = config['data']['pretraining']['validation']
    name, split = selection['dataset'], selection['split']
    protocol = protocol_name(name, selection.get('protocol'))
    path = dataset_path(config, name)
    dataset, _, _ = load_sentence_dataset(path, name, split, selection.get('tgt_lang'), protocol)
    return dataset, {selection_key(name, protocol, split): path}
