import os
import re

session_name_pattern = re.compile(
    r'(_(?P<experiment_name>[a-z]+))?'
    r'_\A.*(?P<subject_id>\d{3})'
    r'_(?P<session_id>\d{1})'
    r'([.](?P<extension>[a-z]{3}))?\Z')


def get_subject_id(name: str) -> str:
    """
    Subject ID will be extracted from file or directory name.
    """
    return name.split('_')[1]


def get_subject_id_from_csv(name: str) -> str:
    """
    Subject ID will be extracted from file or directory name.
    """
    return name.split('_')[0]


def get_session_id(name: str) -> str:
    """
    Session ID will be extracted from file or directory name.
    """
    return name.split('_')[2][0]


def get_session_id_from_csv(name: str) -> str:
    """
    Session ID will be extracted from file or directory name.
    """
    return name.split('_')[1]


def get_experiment_type_from_csv(name: str) -> str:
    """
    Experiment type be extracted from file or directory name. (:-4 to cut .csv)
    """
    return name.split('_')[2][:-4]


def get_session_infos(path: str) -> (str, str, str):
    """
    Get triple of subject_id, session_id

    Infos will be extracted from file or directory name.

    """
    match = session_name_pattern.match(path)

    subject_id = match.group('subject_id')
    session_id = match.group('session_id')
    experiment_name = match.group('experiment_name')

    return subject_id, session_id, experiment_name


def get_session_name(path: str) -> str:
    """
    Get session name from file or directory path.
    """
    return os.path.splitext(os.path.basename(path))[0]


def create_session_name_from_session_info(subject_id: int,
                                          session_id: int):
    """
    Create session name of the form
    {subject_id}_{session_id}

    subject_id will be padded to digits.

    """
    return f'{subject_id:03}_{session_id}'


def create_path_for_results_dir(basepath: str,
                                subject_id: int,
                                session_id: int) -> str:
    """
    Create path of the form {basepath}/{subject_id}_{session_id}

    subject_id will be padded to digits.

    """
    dirname = f'{int(subject_id):03}_{session_id}'
    filepath = os.path.join(basepath, dirname)
    return filepath


def create_filepath_for_csv_file(basepath: str,
                                 subject_id: int,
                                 session_id: int,
                                 experiment_type: str) -> str:
    """
    Create filepath of the form
    {basepath}/{subject_id}_{session_id}_{experiment_type}.csv

    subject_id will be padded to digits.

    """
    filename = f'{int(subject_id):03}_{session_id}_{experiment_type}.csv'
    filepath = os.path.join(basepath, filename)
    return filepath


def create_filepath_for_txt_file(basepath: str,
                                 subject_id: int,
                                 session_id: int,
                                 experiment_type: str) -> str:
    """
    Create filepath of the form
    {basepath}/{subject_id}_{session_id}_{experiment_type}.csv

    subject_id will be padded to digits.

    """
    filename = f'{int(subject_id):03}_{session_id}_{experiment_type}.txt'
    filepath = os.path.join(basepath, filename)
    return filepath


# for indico
# def create_filepath_for_event_file(basepath: str,
#                                    subject_id: int,
#                                    session_id: int,
#                                    text_id: int,
#                                    screen_id: int,
#                                    experiment_type: str) -> str:
#     """
#     Create filepath of the form
#     {basepath}/{subject_id}_{session_id}_{text_id}_{experiment_type}_fixations.csv
#
#     subject_id will be padded to digits.
#
#     """
#     filename = f'{int(subject_id):03}_{session_id}_{int(text_id):02}_{int(screen_id)}_{experiment_type}_fixations.csv'
#     filepath = os.path.join(basepath, filename)
#     return filepath


def create_filepath_for_event_file(
        subj_id: str,
        event_dir: str,
        item_id: str,
        Trial_Index_: str,
        TRIAL_ID: str,
        model: str,
        decoding_strategy: str,
):
    filename = f'{subj_id}_trialid{TRIAL_ID}_{item_id}_trialindex{Trial_Index_}_{model}_{decoding_strategy}_fixations.csv'
    filepath = os.path.join(event_dir, 'event_files')
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath = os.path.join(filepath, filename)
    return filepath


def create_filename_for_rm_file(
        subj_id: str,
        item_id: str,
        Trial_Index_: str,
        TRIAL_ID: str,
        model: str,
        decoding_strategy: str,
):
    filename = f'{subj_id}_trialid{TRIAL_ID}_{item_id}_trialindex{Trial_Index_}_{model}_{decoding_strategy}_reading_measures.csv'
    return filename


def create_filepath_for_asc_file(basepath: str,
                                 subject_id: int,
                                 session_id: int) -> str:
    """
    Create filepath of the form
    {basepath}/{subject_id}_{session_id}.asc

    subject_id will be padded to digits.

    """
    filename = f'{int(subject_id):03}_{session_id}.asc'
    filepath = os.path.join(basepath, filename)
    return filepath
