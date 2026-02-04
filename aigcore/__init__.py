"""
A inclued llm and agent to aigc lib

PyPI: https://pypi.org/project/aigcore/
GitHub: https://github.com/torrentbrave/aigcore
"""

__author__ = "BoHaoChen"

__connect__ = "X @TorrentBrave"

__version__ = "0.0.1"

from ._logger import logger, print
from . import llm
from . import agent


class Null:
    pass


NULL = Null()

__all__ = ["logger", "print", "NULL"]
__all__.extend(llm.__all__ + agent.__all__)
