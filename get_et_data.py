#!/usr/bin/env python

import os
import requests
from argparse import ArgumentParser
import zipfile
from typing import Optional
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
        extract: Optional[bool] = None,
        out_path: Optional[str] = None,
):
    base_url = 'https://osf.io/download/'

    urls = {
        'fixations.csv': 'tk7ph',
        'fixations_corrected.csv': '49qwa',
        'reading_measures_corrected.csv': 'whn7j',
        'stimuli_columns_descriptions.csv': 'je6uz',
        'stimuli.csv': 'p5wf2',
        'participant_info.zip': 'xf67b',
        'subject_level_data.zip': 'rwqs9',
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


def main():
    args = get_parser().parse_args()
    download_data(extract=args.extract, out_path='data')


if __name__ == '__main__':
    raise SystemExit(main())
