__version__ = "0.1.0a"

import logging
import os
import sys
import tomli
import shutil
from pathlib import Path

import matplotlib as mpl

mpl.rcParams["figure.facecolor"] = "white"

PROJECT_PATH = Path(os.path.abspath(__file__)).parent
CONFIG_PATH = Path().home() / ".config" / "QDMpy"
CONFIG_FILE = CONFIG_PATH / "config.ini"
CONFIG_INI = PROJECT_PATH / "config.ini"
DESKTOP = Path().home() / "Desktop"

SRC_PATH = PROJECT_PATH.parent
sys.path.append(SRC_PATH)

### LOGGING ###
from logging.config import fileConfig

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)

logging_conf = Path(PROJECT_PATH, "logging.conf")
fileConfig(logging_conf)

LOG = logging.getLogger("QDMpy")

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

############################### configfile stuff ######################################
def make_configfile(reset: bool = False) -> None:
    """Creates the config file if it does not exist.

    Args:
      reset: bool:  (Default value = False)

    """
    CONFIG_PATH.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists() or reset:
        LOG.info(f"Copying default QDMpy 'config.ini' file to {CONFIG_FILE}")
        shutil.copy2(CONFIG_INI, CONFIG_FILE)


def load_config(file=CONFIG_FILE) -> dict:
    """Loads the config file.

    Args:
        file:  (Default value = CONFIG_FILE)

    Returns:
        dict: Dictionary with the config file contents.
    """
    LOG.info(f"Loading config file: {file}")
    with open(file, "rb") as fileObj:
        return tomli.load(fileObj)


def reset_config():
    """
    Resets the config file.
    """
    make_configfile(reset=True)
    LOG.info("Config file reset")


make_configfile()
SETTINGS = load_config()

############################### CHECK IF pygpufit IS INSTALLED ###############################
import importlib.util

package = "pygpufit"
PYGPUFIT_PRESENT = (
    True if importlib.util.find_spec(package) is not None else False
)  # find_spec will look for the package

if PYGPUFIT_PRESENT is None or sys.platform == "darwin":
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


if __name__ == "__main__":
    LOG.info("This is a module. It is not meant to be run as a script.")
    sys.exit(0)


def test_data_location():
    if sys.platform == "linux":
        return Path("/media/data/Dropbox/FOV18x")
    elif sys.platform == "darwin":
        return Path("/Users/mike/Dropbox/FOV18x")
    elif sys.platform == "win32":
        return Path(r"D:\Dropbox\FOV18x")
    else:
        raise NotImplementedError
