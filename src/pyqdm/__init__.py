__version__ = '0.1.0a'

import logging
import sys
import os
import logging
from logging.config import fileConfig
import tomli  # import tomllib in Python 3.11
import matplotlib as mpl

mpl.rcParams['figure.facecolor'] = 'white'

projectdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(projectdir)
import utils
from utils import set_path, load_config

from logging import getLogger
from logging.config import dictConfig, fileConfig
from pathlib import Path

logging_conf = Path(projectdir, "logging.conf")

fileConfig(logging_conf)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('h5py').setLevel(logging.WARNING)

LOG = logging.getLogger("pyqdm")
LOG.info("WELCOME TO pyqdm")

pyqdm_config = load_config()

desktop = os.path.join(os.path.expanduser("~"), "Desktop")

### CHECK IF pygpufit IS INSTALLED ###
import importlib.util

package = 'pygpufit'
pygpufit_present = importlib.util.find_spec(package)  # find_spec will look for the package
if pygpufit_present is None:
    LOG.error('Can\'t import pyGpufit. The package is necessary for most of the calculations. Functionality of pyqdm '
              'will be greatly diminished.')
    LOG.error(f"try running:\n"
              f">>> pip install --no-index --find-links={os.path.join(projectdir, '../pyGpufit', 'pyGpufit-1.2.0-py2.py3-none-any.whl')} pyGpufit")
else:
    import pygpufit.gpufit as gf

    LOG.info(f'CUDA available: {gf.cuda_available()}')
    LOG.info("CUDA versions runtime: {}, driver: {}".format(*gf.get_cuda_version()))

if __name__ == '__main__':
    LOG.info("This is a module. It is not meant to be run as a script.")
    sys.exit(0)
