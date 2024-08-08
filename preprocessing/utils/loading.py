import os
import yaml
from typing import Optional

def load_config(path: Optional[str] = None):
    if path is None:
        path = os.path.abspath('preprocessing/config.yaml')
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

