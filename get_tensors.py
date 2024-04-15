#!/usr/bin/env python

import requests


def main():
    # Define DOI of the dataset
    doi = 'https://doi.org/10.7910/DVN/GCU0W8'

    # Dataverse API endpoint
    api_url = f'https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:{doi}'

    # Make GET request to retrieve metadata
    response = requests.get(api_url)

    # Check if request was successful
    if response.status_code == 200:
        # Parse JSON response
        metadata = response.json()

        # Extract download links for files
        files = metadata['data']['latestVersion']['files']
        for file_info in files:
            file_id = file_info['dataFile']['id']
            download_url = f'https://dataverse.harvard.edu/api/access/datafile/{file_id}'

            # Download file
            file_response = requests.get(download_url)
            # Save file to disk or process it as needed
            with open(file_info['dataFile']['filename'], 'wb') as f:
                f.write(file_response.content)
    else:
        print('Failed to retrieve dataset metadata')

if __name__ == '__main__':
    raise SystemExit(main())


