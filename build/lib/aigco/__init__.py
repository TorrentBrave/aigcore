"""
A inclued llm and agent to aigc lib

PyPI: https://pypi.org/project/aigco/
GitHub: https://github.com/torrentbrave/aigco
"""

__author__ = "BoHaoChen"

__connect__ = "X @TorrentBrave"

__version__ = "0.0.2"

from ._logger import logger, print
from . import llm
from . import inference
# from . import agent


class Null:
    pass


NULL = Null()

__all__ = ["logger", "print", "NULL", "llm", "inference"]
# __all__.extend(llm.__all__)
