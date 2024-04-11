import os
import yaml

# TODO is used
def load_config(path: str = None):
    if path is None:
        path = os.path.abspath('preprocessing/config.yaml')
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_filepaths(dirpath: os.path, extension: str = None):
    path_list = [os.path.join(dirpath, filename)
                 for filename in os.listdir(dirpath)
                 if (extension is None)
                 or os.path.splitext(filename)[1] == '.' + extension]
    path_list.sort()
    return path_list
