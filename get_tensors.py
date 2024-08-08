#!/usr/bin/env python

import requests
import json
import os

from typing import List, Union, Any
from argparse import ArgumentParser 

def get_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        choices=['mistral', 'phi2', 'wizardlm'],
        help='download the tensors only from the specified model(s). If not provided, they will be downloaded for all models.',
        nargs='+',
    )
    parser.add_argument(
        '--dec',
        type=str,
        choices=['beam_search', 'greedy_search', 'sampling', 'topk', 'topp'],
        help='download the tensors only from the specified decoding strategy/strategies. If not provided, they will be downloaded for all decoding strategies.',
        nargs='+',
    )
    parser.add_argument(
        '--tensor',
        type=str,
        choices=['attentions', 'beam_indices', 'hidden_states', 'sequences', 'scores'],
        help='download only the specified tensors. If not provided, all tensors will be downloaded.',
        nargs='+',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='the directory into which to save the tensors.',
    )
    return parser.parse_args()


def string_in_filename(strings: Union[List[str], Any], filename: str):
    for string in strings:
        if string in filename:
            return True
    else:
        return False


def main():

    # get argparse args
    args = get_args()
    models = args.model
    decs = args.dec
    tensors = args.tensor
    out_dir = args.output_dir

    # Dataverse API endpoint
    api_url = 'https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GCU0W8'

    # Make GET request to retrieve metadata
    response = requests.get(api_url)

    with open('tensor_data/metadata/metadata.json', 'r') as f:
        metadata = json.load(f)

    if not out_dir:
        output_dir = 'tensor_data/tensors/'
    else:
        output_dir = out_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if response.status_code == 200:
        files = metadata['datasetVersion']['files']
        for file_info in files:
            file_id = file_info['dataFile']['id']
            filename = file_info['dataFile']['filename']
            subdirectory = file_info['directoryLabel']

            # check if arguments provided to subset the tensor dataset
            if tensors:
                download = string_in_filename(strings=tensors, filename=filename)
                if not download:
                    continue
            if decs:
                download = string_in_filename(strings=decs, filename=filename)
                if not download:
                    continue
            if models:
                download = string_in_filename(strings=models, filename=filename)
                if not download:
                    continue

            download_url = f'https://dataverse.harvard.edu/api/access/datafile/{file_id}'
            # Download file
            file_response = requests.get(download_url)
            # Save file to disk
            save_dir = os.path.join(output_dir, subdirectory)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(os.path.join(save_dir, file_info['dataFile']['filename']), 'wb') as f:
                f.write(file_response.content)
            
            print(f'--- downloaded {filename}')

    else:
        print('Failed to retrieve dataset metadata')



if __name__ == '__main__':
    raise SystemExit(main())


