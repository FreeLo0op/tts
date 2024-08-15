import functools
import logging

__all__ = [
    'logger',
]

class Logger(object):
    def __init__(self, name: str=None):
        name = '---Tal Frontend---' if not name else name
        self.logger = logging.getLogger(name)

        log_config = {
            'DEBUG': 10,
            'INFO': 20,
            'TRAIN': 21,
            'EVAL': 22,
            'WARNING': 30,
            'ERROR': 40,
            'CRITICAL': 50,
            'EXCEPTION': 100,
        }
        for key, level in log_config.items():
            logging.addLevelName(level, key)
            method = self.logger.exception if key == 'EXCEPTION' else functools.partial(self.__call__, level)
            setattr(self, key.lower(), method)

        self.format = logging.Formatter(
            fmt='[%(asctime)-15s] [%(levelname)8s] - %(message)s')

        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.format)

        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.ERROR)
        self.logger.propagate = False

        # 捕获警告并设置日志级别为ERROR
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").setLevel(logging.ERROR)

    def __call__(self, log_level: str, msg: str):
        self.logger.log(log_level, msg)

logger = Logger()