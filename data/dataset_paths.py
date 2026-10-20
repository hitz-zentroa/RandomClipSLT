"""Documented dataset layouts, with optional config and CLI path overrides."""

from pathlib import Path


def resolve_path(config, dataset, key, default, override=None, split=None, must_exist=False):
    settings = config['data']['datasets'][dataset]
    location = f'data.datasets.{dataset}'
    if split is not None:
        settings_for_path = settings.get('sources', {}).get(split, {})
        location += f'.sources.{split}'
    else:
        settings_for_path = settings
    explicit = override or settings_for_path.get(key)
    if explicit:
        path = Path(explicit).expanduser()
    else:
        if not settings.get('root'):
            raise ValueError(f'Set data.datasets.{dataset}.root or {location}.{key}.')
        path = Path(settings['root']).expanduser() / default
    if must_exist and not path.is_file():
        raise FileNotFoundError(
            f'Missing {dataset} file: {path}. Follow the documented dataset layout '
            f'or override {location}.{key}.'
        )
    return path
