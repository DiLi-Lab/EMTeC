import os
import re

session_name_pattern = re.compile(
    r'(_(?P<experiment_name>[a-z]+))?'
    r'_\A.*(?P<subject_id>\d{3})'
    r'_(?P<session_id>\d{1})'
    r'([.](?P<extension>[a-z]{3}))?\Z')


def create_filepath_for_event_file(
        subj_id: str,
        event_dir: str,
        item_id: str,
) -> str:
    filename = f'{subj_id}-{item_id}-fixations.csv'
    filepath = os.path.join(event_dir, 'event_files')
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath = os.path.join(filepath, filename)
    return filepath
