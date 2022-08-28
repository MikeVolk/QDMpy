import os
from logging import getLogger
from logging.config import fileConfig

import QDMpy

fileConfig(os.path.join(QDMpy.projectdir, "logging.conf"))
LOG = getLogger("QDMpy")
