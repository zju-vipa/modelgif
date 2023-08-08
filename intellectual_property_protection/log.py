import logging
import os
import sys


class NoOp:
    def __getattr__(self, *args):
        def no_op(*args, **kwargs):
            """Accept every signature by doing non-operation."""
            pass

        return no_op


def get_logger(log_dir, log_name=None, resume="", is_rank0=True):
    """Get the program logger.
    Args:
        log_dir (str): The directory to save the log file.
        log_name (str, optional): The log filename. If None, it will use the main
            filename with ``.log`` extension. Default is None.
        resume (boolean): If False, open the log file in writing and reading mode.
            Else, open the log file in appending and reading mode; Default is "".
        is_rank0 (boolean): If True, create the normal logger; If False, create the null
           logger, which is useful in DDP training. Default is True.
    """
    if is_rank0:
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)

        # StreamHandler
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        logger.addHandler(stream_handler)

        # FileHandler
        mode = "a+" if resume != "" else "w+"
        if log_name is None:
            log_name = os.path.basename(sys.argv[0]).split(".")[0] + (".log")
        file_handler = logging.FileHandler(os.path.join(log_dir, log_name), mode=mode)
        file_handler.setLevel(level=logging.INFO)
        logger.addHandler(file_handler)
    else:
        logger = NoOp()

    return logger
