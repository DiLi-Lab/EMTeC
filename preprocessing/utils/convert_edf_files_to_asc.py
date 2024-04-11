#!/usr/bin/env python

import concurrent.futures
import logging
import stat
import subprocess
import os

from typing import List
from tqdm import tqdm
from argparse import ArgumentParser

from preprocessing.utils.loading import load_config


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    # parser.add_argument(
    #     '--dec-v',
    #     type=str,
    #     nargs='+',
    #     default=['decoding_v15_deploy', 'decoding_v16_deploy', 'decoding_v17_deploy', 'decoding_v18_deploy',
    #              'decoding_v19_deploy'],
    # )
    parser.add_argument(
        '--path-to-data',
        type=str,
        default='data/',
    )
    return parser


def convert_edf_to_asc_file(edf_filepath: str,
                            asc_filepath: str,
                            exclude_subjects: List[str],
                            write_logs: bool = True,
                            skip_existing: bool = True):
    """
    Converts edf file to asc file using the edf2asc tool from
    SR-Research.

    Writes logs to directory holding the edf file.

    """
    if skip_existing and os.path.isfile(asc_filepath):
        print(f'---{asc_filepath} already exists. skipping.')
        return

    # exclude subjects
    subj_id = asc_filepath.split('/')[-1][:-4]
    if subj_id in exclude_subjects:
        print(f'---excluding subject {subj_id}')
        return

    p = subprocess.Popen(['preprocessing/edf2asc', edf_filepath, asc_filepath],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    print("wow", p)
    stdout, stderr = p.communicate()
    file_permissions = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP
    os.chmod(asc_filepath, file_permissions)

    if write_logs:
        edf_dirpath = os.path.dirname(edf_filepath)

        if stdout:
            stdout_filename = 'edf2asc.log'
            stdout_filepath = os.path.join(edf_dirpath, stdout_filename)
            stdout_file = open(stdout_filepath, "wb")
            stdout_file.write(stdout.replace(b'\r', b'\n'))
            stdout_file.close()

        if stderr:
            stderr_filename = 'edf2asc-error.log'
            stderr_filepath = os.path.join(edf_dirpath, stderr_filename)
            stderr_file = open(stderr_filepath, "wb")
            stderr_file.write(stderr.replace(b'\r', b'\n'))
            stderr_file.close()

    success_string_prefix = b'Converted successfully: '
    last_line = stdout.splitlines()[-1]
    if not last_line.startswith(success_string_prefix):
        raise Exception(last_line)

    if stderr:
        raise Exception(stderr)


def main():
    """
    Converts edf file in each session directory of experiment results
    path (specified in config) to respective asc file.
    """

    args = get_parser().parse_args()

    config = load_config()

    # load which subjects to exclude, and which screens of specific subjects to exclude
    results_base_path = config['results_base_path']
    exclude_subjects = config['exclude']['subjects']

    # load parameters for the edf to asc conversion
    n_jobs = config['edf2asc']['n_jobs']
    write_logs = config['edf2asc']['write_logs']
    skip_existing = config['edf2asc']['skip_existing']

    data_basepath = args.path_to_data
    subj_dirs = os.listdir(data_basepath)

    if '.DS_Store' in subj_dirs:
        subj_dirs.remove('.DS_Store')

    def prepare_and_convert(subj_dir: str):
        """
        Prepares the filepaths for conversion and eventually convert.
        The function is nested to simplify accessing base paths.
        :param subj_dir:
        :return:
        """

        edf_filename = subj_dir + '.edf'
        edf_filepath = os.path.join(data_basepath, subj_dir, edf_filename)

        asc_filename = subj_dir + '.asc'
        asc_filepath = os.path.join(data_basepath, subj_dir, asc_filename)

        print("edf_filepath ", edf_filepath)
        print("asc_filepath ", asc_filepath)
        print("write_logs ", write_logs)
        print("skip_existing ", skip_existing)
        print("-----------------")

        # convert
        convert_edf_to_asc_file(
            edf_filepath=edf_filepath,
            asc_filepath=asc_filepath,
            exclude_subjects=exclude_subjects,
            write_logs=write_logs,
            skip_existing=skip_existing,
        )

        return asc_filename

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(tqdm(executor.map(prepare_and_convert,
                                         subj_dirs),
                            total=len(subj_dirs)))
        results.sort()





    #
    #
    # for dec_folder in dec_folders:
    #
    #     results_subj_path = results_base_path.format(dec_folder=dec_folder)
    #     # results_dirpath = os.path.join('data', dec_folder, 'results')
    #     asc_dirpath = results_subj_path
    #
    #     # results_dirpath = config['experiment_results_path']
    #     # asc_dirpath = config['eyetracking_asc_path']
    #     n_jobs = config['edf2asc']['n_jobs']
    #     write_logs = config['edf2asc']['write_logs']
    #     skip_existing = config['edf2asc']['skip_existing']
    #     filename_map = config['edf2asc']['filename_mapping']
    #
    #     def prepare_and_convert(session_dirname: str):
    #         """
    #         Prepares filepaths for conversion and eventually convert.
    #
    #         Function is nested to simplify access basepaths.
    #
    #         """
    #         edf_filename = session_dirname + '.edf'
    #         edf_filepath = os.path.join(results_subj_path,
    #                                     session_dirname,
    #                                     edf_filename)
    #
    #         black_list = [
    #
    #         ]
    #
    #         # if session_dirname in black_list:
    #         #     print('Skipping blacklisted session: {}'.format(session_dirname))
    #         #     return ''
    #
    #         # Adjust asc_filename for incorrect session names
    #         # try:
    #         #     new_session_dirname = filename_map[session_dirname]
    #         #     print('replace {} with {}'.format(session_dirname, new_session_dirname))
    #         #     session_dirname = new_session_dirname
    #         # except KeyError:
    #         #     pass
    #
    #         asc_filename = session_dirname + '.asc'
    #         asc_filepath = os.path.join(asc_dirpath, session_dirname, asc_filename)
    #         print("edf_filepath ", edf_filepath)
    #         print("asc_filepath ", asc_filepath)
    #         print("write_logs ", write_logs)
    #         print("skip_existing ", skip_existing)
    #         print("-----------------")
    #
    #         # convert_edf_to_asc_file(edf_filepath, asc_filepath,
    #         #                         write_logs, skip_existing)
    #         convert_edf_to_asc_file(
    #             edf_filepath=edf_filepath,
    #             asc_filepath=asc_filepath,
    #             exclude_subjects=exclude_subjects,
    #             write_logs=write_logs,
    #             skip_existing=skip_existing,
    #         )
    #         # try:
    #         #     convert_edf_to_asc_file(
    #         #         edf_filepath=edf_filepath,
    #         #         asc_filepath=asc_filepath,
    #         #         write_logs=write_logs,
    #         #         skip_existing=skip_existing,
    #         #     )
    #         # except Exception as e:
    #         #     print( asc_filename, e)
    #         #     return ''
    #         return asc_filename
    #
    #     session_dirnames = os.listdir(results_subj_path)  # ET_01, ET_02, ...
    #     session_dirnames.sort()
    #
    #     if '.DS_Store' in session_dirnames:
    #         session_dirnames.remove('.DS_Store')
    #
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
    #         results = list(tqdm(executor.map(prepare_and_convert,
    #                                          session_dirnames),
    #                             total=len(session_dirnames)))
    #         results.sort()


if __name__ == "__main__":
    main()
