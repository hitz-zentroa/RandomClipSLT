import h5py
import numpy as np
import webvtt

from bs4 import BeautifulSoup


CAPTION_DTYPE = np.dtype([
    ('sentence', h5py.string_dtype(encoding='utf-8')),
    ('start_time', np.int32),
    ('end_time', np.int32)
])


def clean_caption_text(caption_text):
    return BeautifulSoup(
        BeautifulSoup(caption_text, 'lxml').text,
        'lxml'
    ).text.replace('\n', ' ')


def get_caption_time_ms(caption_time):
    hours, minutes, seconds, milliseconds = caption_time.to_tuple()
    return (
        hours * 3_600_000
        + minutes * 60_000
        + seconds * 1000
        + milliseconds
    )


def read_caption_file(path):
    captions = []

    for caption in webvtt.read(path):
        captions.append({
            'sentence': clean_caption_text(caption.text),
            'start_time': get_caption_time_ms(caption.start_time),
            'end_time': get_caption_time_ms(caption.end_time)
        })

    return captions


def caption_rows(captions):
    return [
        (caption['sentence'], caption['start_time'], caption['end_time'])
        for caption in captions
    ]


def dataset_caption_rows(captions_ds):
    return [
        (
            row['sentence'].decode('utf-8')
            if isinstance(row['sentence'], bytes)
            else row['sentence'],
            int(row['start_time']),
            int(row['end_time'])
        )
        for row in captions_ds[:]
    ]


def create_caption_dataset(
    captions_group,
    target_lang,
    captions,
    filename,
    is_synthetic=False
):
    captions_array = np.array(caption_rows(captions), dtype=CAPTION_DTYPE)
    captions_ds = captions_group.create_dataset(target_lang, data=captions_array)
    captions_ds.attrs['is_synthetic'] = is_synthetic
    captions_ds.attrs['filename'] = filename
    return captions_ds
