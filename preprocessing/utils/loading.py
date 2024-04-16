import os
import yaml

def load_config(path: str = None):
    if path is None:
        path = os.path.abspath('preprocessing/config.yaml')
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

