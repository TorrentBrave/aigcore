import os
import datetime
import logging
from dotenv import load_dotenv

load_dotenv()


__all__ = ["logger", "print"]

_print = print


def print(
    *args,
    **kwargs,
) -> None:
    """
    Prints, and then flushes instantly.

    The usage is the same as the built-in `print`.

    Parameters
    ----------
    See also the built-in `print`.

    Returns
    -------
    None

    Notes
    -----
    `args` and `kwargs` are passed to the built-in `print`. `flush` is
    overridden to True no matter what.
    """
    kwargs["flush"] = True
    _print(*args, **kwargs)


def logger(
    *,
    name: str | None = None,
    dir: str | None = None,
) -> logging.Logger:
    """
    Returns a pre-configured `logging.Logger` object.

    INFO logs are written to both the .log file and the console.

    WARNING logs are written to the console only.

    Parameters
    ----------
    name: str | None = None
        `logging.Logger.name`. If *None*, it is set to 'wqb' followed by
        the current datetime. The filename of the .log file is set to
        `name` followed by '.log'.
        Specifying a name is required if you want to prevent multiple log files and keep everything in a single trace.
    dir: str | None = "logs"
        The directory where the .log file will be stored.
        Defaults to 'logs'. If the directory does not exist, it will be created.


    Returns
    -------
    logging.Logger
        A pre-configured `logging.Logger` object.
    """
    if dir is None:
        dir = os.getenv("LOGDIR", "logs")

    if name is None:
        name = "aigco" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    if dir is not None:
        os.makedirs(dir, exist_ok=True)
        log_path = os.path.join(dir, f"{name}.log")
    else:
        log_path = f"{name}.log"

    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)
    handler1 = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler1.setLevel(logging.INFO)
    handler1.setFormatter(
        logging.Formatter(fmt="# %(levelname)s %(asctime)s\n%(message)s\n")
    )
    logger.addHandler(handler1)
    handler2 = logging.StreamHandler()
    handler2.setLevel(logging.WARNING)
    handler2.setFormatter(
        logging.Formatter(fmt="# %(levelname)s %(asctime)s\n%(message)s\n")
    )
    logger.addHandler(handler2)
    return logger
