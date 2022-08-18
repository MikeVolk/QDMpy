from logging import getLogger
from logging.config import dictConfig, fileConfig
import pyqdm
import os
fileConfig(os.path.join(pyqdm.projectdir, "logging.conf"))
LOG = getLogger("pyqdm")