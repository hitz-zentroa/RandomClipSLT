"""Dataset selection and a common reader for sentence keypoint files."""

import importlib

import h5py


DATASETS = {
    'how2sign': {'splits': ('train', 'val', 'test'), 'source': 'ase', 'targets': ('en',)},
    'lse_health': {'splits': ('train', 'val', 'test'), 'source': 'ssp', 'targets': ('es',)},
    'flores_lse': {'splits': ('dev', 'devtest'), 'source': 'ssp',
                   'targets': ('en', 'es', 'ca', 'eu', 'gl')},
}


def dataset_options(name, split, target=None):
    options = DATASETS[name]
    if split not in options['splits']:
        raise ValueError(f'{name}: split must be one of {options["splits"]}.')
    target = target or options['targets'][0]
    if target not in options['targets']:
        raise ValueError(f'{name}: target must be one of {options["targets"]}.')
    return options['source'], target


def load_segments(name, args, config):
    dataset_options(name, args.split)
    return importlib.import_module(f'data.{name}').load_segments(args, config)


def load_evaluation_dataset(path, name, split, target=None):
    from datasets import Dataset

    source, target = dataset_options(name, split, target)
    instances, manifest = [], []
    with h5py.File(path, 'r') as file:
        metadata = {key: value.item() if hasattr(value, 'item') else value
                    for key, value in file.attrs.items()}
        # Old How2Sign and FLORES-LSE files have no root metadata.
        if 'schema_version' in metadata or name == 'lse_health':
            if (not metadata.get('complete', False)
                    or metadata.get('dataset') != name or metadata.get('split') != split):
                raise ValueError('Incomplete HDF5 file or dataset/split mismatch.')
            if metadata.get('expected_segments') != len(file):
                raise ValueError('HDF5 segment count does not match extraction metadata.')
        if name == 'lse_health' and metadata.get('split_protocol') != 'SD&TD':
            raise ValueError('LSE-Health requires the SD&TD protocol.')
        ids = list(file.keys())
        if 'schema_version' in metadata:
            ids.sort(key=lambda key: int(file[key].attrs['order']))
        for sentence_id in ids:
            group = file[sentence_id]
            if ('split' in group.attrs and group.attrs['split'] != split
                    or name == 'flores_lse' and 'split' not in group.attrs):
                raise ValueError(f'{sentence_id}: unexpected or missing split.')
            if group['keypoints'].shape[0] == 0:
                raise ValueError(f'{sentence_id}: empty keypoints.')
            reference_key = f'text_{target}' if name == 'flores_lse' else 'sentence'
            instances.append({
                'instance_type': 'sentence_level_dataset_slt',
                'src_lang': source, 'tgt_lang': target, 'dataset_name': f'{name}_{split}',
                'tgt_sentence': group.attrs[reference_key],
                'height': group.attrs['height'], 'width': group.attrs['width'],
                'sentence_id': sentence_id,
            })
            manifest.append({'sentence_id': sentence_id, **{
                key: value.item() if hasattr(value, 'item') else value
                for key, value in group.attrs.items()
            }})
    if not instances:
        raise ValueError(f'No sentences found in {path}.')
    return Dataset.from_list(instances), manifest, metadata
