import os
from logging import getLogger
from logging.config import fileConfig

import pyqdm

fileConfig(os.path.join(pyqdm.projectdir, "logging.conf"))
LOG = getLogger("pyqdm")