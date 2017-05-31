import logging

__all__ = ['get_path', 'init_logger', 'load_config', 'disable_other_log']


def get_path(name):
    """
    Get relative path of this project
    :param name: 
    :return: 
    """
    import os
    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, name))
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
COLORS = {
    'WARNING': YELLOW,
    'INFO': WHITE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED,
    'RED': RED,
    'GREEN': GREEN,
    'YELLOW': YELLOW,
    'BLUE': BLUE,
    'MAGENTA': MAGENTA,
    'CYAN': CYAN,
    'WHITE': WHITE,
}


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg):
        logging.Formatter.__init__(self, msg)

    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        message = logging.Formatter.format(self, record)
        message = message.replace("$RESET", RESET_SEQ) \
            .replace("$BOLD", BOLD_SEQ)
        for k, v in COLORS.items():
            message = message.replace("$" + k, COLOR_SEQ % (v + 30)) \
                .replace("$BG" + k, COLOR_SEQ % (v + 40)) \
                .replace("$BG-" + k, COLOR_SEQ % (v + 40))
        return message + RESET_SEQ


def init_logger(name):
    """
    Initialize a logger with certain name
    :param name: logger name
    :return: logger
    :rtype: logging.Logger
    """
    import logging
    import logging.handlers
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = 0
    _nf = ['[%(asctime)s]',
           '[%(name)s]',
           '[%(filename)s:%(funcName)s:%(lineno)d]',
           '[%(levelname)s]',
           '[%(message)s]']
    _cf = ['$GREEN[%(asctime)s]$RESET',
           '[%(name)s]',
           '$BLUE[%(filename)s:%(funcName)s:%(lineno)d]$RESET',
           '[%(levelname)s]',
           '$CYAN[%(message)s]$RESET']
    nformatter = logging.Formatter('-'.join(_nf))
    cformatter = ColoredFormatter('-'.join(_cf))

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(cformatter)

    rf = logging.handlers.RotatingFileHandler(get_path('log') + '/' + name + '.log',
                                              maxBytes=1 * 1024 * 1024,
                                              backupCount=5)
    rf.setLevel(logging.DEBUG)
    rf.setFormatter(nformatter)

    logger.addHandler(ch)
    logger.addHandler(rf)
    return logger


def load_config(name):
    import yaml
    with open(get_path('config') + '/' + name) as f:
        return yaml.load(f)


def disable_other_log():
    import os, gym
    gym.undo_logger_setup()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
