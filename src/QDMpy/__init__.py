__version__ = "0.1.0a"

import logging
import os
import sys
from logging.config import fileConfig
from pathlib import Path

import matplotlib as mpl

mpl.rcParams["figure.facecolor"] = "white"

PROJECT_PATH = Path(os.path.abspath(__file__)).parent
CONFIG_PATH = Path().home() / ".config" / "QDMpy"
CONFIG_FILE = CONFIG_PATH / "config.ini"
CONFIG_INI = PROJECT_PATH / "config.ini"

SRC_PATH = PROJECT_PATH.parent
sys.path.append(SRC_PATH)

logging_conf = Path(projectdir, "logging.conf")

logging_conf = Path(PROJECT_PATH, "logging.conf")

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)

LOG = logging.getLogger(f"QDMpy")

import coloredlogs

coloredlogs.install(
    level="DEBUG",
    fmt="%(asctime)s %(levelname)8s %(name)s.%(funcName)s >> %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    logger=LOG,
    isatty=True,
)

LOG.info("WELCOME TO QDMpy")
LOG.debug(f"QDMpy version {__version__} installed at {PROJECT_PATH}")
LOG.debug(f"QDMpy config file {CONFIG_FILE}")

make_configfile()
settings = load_config()

desktop = os.path.join(os.path.expanduser("~"), "Desktop")

### CHECK IF pygpufit IS INSTALLED ###
import importlib.util

package = "pygpufit"
PYGPUFIT_PRESENT = importlib.util.find_spec(package)  # find_spec will look for the package

if PYGPUFIT_PRESENT is None:
    LOG.error(
        "Can't import pyGpufit. The package is necessary for most of the calculations. Functionality of QDMpy "
        "will be greatly diminished."
    )
    LOG.error(
        f"try running:\n"
        f">>> pip install --no-index --find-links={os.path.join(SRC_PATH, 'pyGpufit', 'win', 'pyGpufit-1.2.0-py2.py3-none-any.whl')} pyGpufit"
    )
else:
    import pygpufit.gpufit as gf

    LOG.info(f"CUDA available: {gf.cuda_available()}")
    LOG.info("CUDA versions runtime: {}, driver: {}".format(*gf.get_cuda_version()))

from QDMpy.core.fit import Fit
from QDMpy.core.odmr import ODMR
from QDMpy.core.qdm import QDM

if __name__ == "__main__":
    LOG.info("This is a module. It is not meant to be run as a script.")
    sys.exit(0)
