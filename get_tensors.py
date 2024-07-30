#!/usr/bin/env python

import requests
import json
import os


def main():

    # Dataverse API endpoint
    api_url = 'https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/GCU0W8'

    # Make GET request to retrieve metadata
    response = requests.get(api_url)

    with open('tensor_data/metadata/metadata.json', 'r') as f:
        metadata = json.load(f)

    output_dir = 'tensor_data/tensors/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if response.status_code == 200:
        files = metadata['datasetVersion']['files']
        for file_info in files:
            file_id = file_info['dataFile']['id']
            download_url = f'https://dataverse.harvard.edu/api/access/datafile/{file_id}'
            # Download file
            file_response = requests.get(download_url)
            # Save file to disk or process it as needed
            with open(os.path.join(output_dir, file_info['dataFile']['filename']), 'wb') as f:
                f.write(file_response.content)
            breakpoint()

    else:
        print('Failed to retrieve dataset metadata')



if __name__ == '__main__':
    raise SystemExit(main())


