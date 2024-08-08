#!/usr/bin/env python
from __future__ import annotations

import os
import requests
from argparse import ArgumentParser
import zipfile
from tqdm import tqdm


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--extract',
        action='store_true',
        help='Extract the downloaded zip folders',
    )
    return parser


def download_data(
        extract: bool | None  = None,
        out_path: str | None = None,
):
    base_url = 'https://osf.io/download/'

    urls = {
        'fixations.csv': '2hs8p',
        'fixations_corrected.csv': 'w3gan',
        'reading_measures.csv': 's4ny8',
        'reading_measures_corrected.csv': 'wa3ty',
        'stimuli_columns_descriptions.csv': 'tpr5e',
        'stimuli.csv': 'vgp9a',
        'participant_info.zip': '7mw6u',
        'subject_level_data.zip': '374sk',
    }

    for data, resource in (pbar := tqdm(urls.items())):
        pbar.set_description(f'Downloading {"and extracting " if extract else ""}{data}')

        # downloading the file by sending the request to the URL
        url = base_url + resource
        req = requests.get(url, stream=True)

        if not os.path.exists(out_path):
            os.makedirs(out_path)
        path_to_file = os.path.join(out_path, data)

        with open(path_to_file, 'wb') as outfile:
            for chunk in req.iter_content():
                outfile.write(chunk)

        if extract and data.endswith('.zip'):
            with zipfile.ZipFile(path_to_file, 'r') as zip_ref:
                zip_ref.extractall(out_path)


def main() -> int:
    args = get_parser().parse_args()
    download_data(extract=args.extract, out_path='data')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
