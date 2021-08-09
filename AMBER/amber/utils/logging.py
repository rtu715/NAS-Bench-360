import logging
import os


def setup_logger(working_dir='.', verbose_level=logging.INFO):
    """The logging used by throughout the training envrionment

    Parameters
    ----------
    working_dir : str
        File path to working directory. Logging will be stored in working directory.

    verbose_level : int
        Verbosity level; can be specified as in ``logging``

    Returns
    -------
    logger : the logging object
    """
    # setup logger
    logger = logging.getLogger('AMBER')
    logger.setLevel(verbose_level)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(working_dir, 'log.AMBER.txt'))
    fh.setLevel(verbose_level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(verbose_level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -\n %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger