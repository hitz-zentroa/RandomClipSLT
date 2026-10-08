import argparse
import csv
import glob
import h5py
import os
import yaml

from .caption_utils import create_caption_dataset, read_caption_file


def find_hdf5_path(hdf5_root, sign_lang):
    hdf5_paths = glob.glob(os.path.join(hdf5_root, sign_lang, '*.h5'))

    if not hdf5_paths:
        raise FileNotFoundError(
            f'HDF5 file not found for sign language: {sign_lang}'
        )

    if len(hdf5_paths) > 1:
        raise RuntimeError(
            f'Multiple HDF5 files found for sign language {sign_lang}: '
            f'{", ".join(hdf5_paths)}'
        )

    return hdf5_paths[0]


def add_gloss_caption_file(caption_filename, sign_lang, config):
    captions_path = os.path.join(
        config['data']['captions']['path'],
        sign_lang,
        caption_filename
    )
    if not os.path.isfile(captions_path):
        raise FileNotFoundError(f'Caption file not found: {captions_path}')

    hdf5_path = find_hdf5_path(config['data']['hdf5_path'], sign_lang)
    video_id = os.path.basename(caption_filename).split('.')[0]
    target_lang = f'{sign_lang}_gloss'

    with h5py.File(hdf5_path, 'a') as hdf5_file:
        if video_id not in hdf5_file:
            raise KeyError(
                f'Video ID {video_id} not found in HDF5 file for sign '
                f'language {sign_lang}: {hdf5_path}'
            )

        video_group = hdf5_file[video_id]
        captions_group = video_group.require_group('captions')

        if target_lang in captions_group:
            raise RuntimeError(
                f'Captions already exist for video ID {video_id} in '
                f'language {target_lang}'
            )

        captions = read_caption_file(captions_path)
        create_caption_dataset(
            captions_group,
            target_lang,
            captions,
            filename=caption_filename
        )

    print(
        f'Added {caption_filename} to video {video_id} as {target_lang}.'
    )


def main(tsv_path, config):
    with open(tsv_path, newline='', encoding='utf-8') as tsv_file:
        rows = csv.reader(tsv_file, delimiter='\t')

        for line_number, row in enumerate(rows, start=1):
            if not row or all(not value.strip() for value in row):
                continue

            if len(row) != 2:
                raise ValueError(
                    f'Expected 2 TSV columns on line {line_number}, '
                    f'found {len(row)}'
                )

            caption_filename, sign_lang = (
                value.strip()
                for value in row
            )
            if not caption_filename or not sign_lang:
                raise ValueError(
                    f'Empty caption filename or sign language on '
                    f'line {line_number}'
                )

            add_gloss_caption_file(caption_filename, sign_lang, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Add manually selected gloss caption files to HDF5 files.'
    )
    parser.add_argument('tsv_path')
    parser.add_argument(
        '--config-path',
        type=str,
        required=True
    )
    args = parser.parse_args()

    with open(args.config_path, encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    main(args.tsv_path, config)
