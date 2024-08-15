import logging
from logging.handlers import TimedRotatingFileHandler, RotatingFileHandler
from threading import Lock
import os
import datetime
import json
import time
PROJECT_NAME = "tal_vits"  #git工程名称
LOG_TYPE = "python" # 日志类别 语言
LOG_PATH = "/home/logs/xeslog/"
# LOG_PATH = "tmp/"
#DEBUG < INFO < WARNING < ERROR < CRITICAL，而日志的信息量是依次减少


class LoggerProject(object):

    def __init__(self):
        self.mutex = Lock()
        self.formatter = ''

        # self.formatter = '%(asctime)s | %(process)s | %(pathname)s[line:%(lineno)d] | %(funcName)s | %(levelname)s | %(message)s'

    def _create_logger(self, level):
        _logger = logging.getLogger(__name__)
        _logger.setLevel(level)
        return _logger

    def _file_logger(self, level, port):
        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)
        log_file = LOG_PATH + PROJECT_NAME + '-' + str(
            port) + '.log'
        size_rotate_file = RotatingFileHandler(filename=log_file, backupCount=3, encoding='utf-8', maxBytes=1024*1024*1024)
        # size_rotate_file = TimedRotatingFileHandler(filename=log_file, when='MIDNIGHT', backupCount=1,
        #                                             encoding='utf-8')  # MIDNIGHT
        size_rotate_file.setFormatter(logging.Formatter(self.formatter))
        size_rotate_file.setLevel(level)
        return size_rotate_file

    def _console_logger(self, level):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(self.formatter))
        return console_handler

    def pub_logger(self, level, port=0):
        logger = self._create_logger(level)
        self.mutex.acquire()
        logger.addHandler(self._file_logger(level, port))
        logger.addHandler(self._console_logger(level))
        self.mutex.release()
        return logger


class Log(object):
    def __init__(self, level=logging.INFO, port=0, debug=False):
        log_pro = LoggerProject()
        self.logger = log_pro.pub_logger(level, port)
        self.debug = debug
        self.logging_collection = {logging.DEBUG: "DEBUG", logging.INFO: "INFO", logging.WARNING: "WARNING",
                                   logging.ERROR: "ERROR"}
        self.func_collection = {
            logging.DEBUG: self.logger.debug,
            logging.INFO: self.logger.info,
            logging.WARNING: self.logger.warning,
            logging.ERROR: self.logger.error
        }
    
    def info(self, code, message, trace_id):
        msg =  {
            "level": "INFO",
            "time": str(datetime.datetime.now()),
            "trace_id": trace_id,
            "type": LOG_TYPE,
            "project": PROJECT_NAME,
            "content": {
                "code": code,
                "message": message
            }
        }
        self.logger.info(json.dumps(msg, ensure_ascii=False))
        
    def error(self, code, message, trace_id, exception=None):
        msg =  {
            "level": "ERROR",
            "time": str(datetime.datetime.now()),
            "trace_id": trace_id,
            "type": LOG_TYPE,
            "project": PROJECT_NAME,
            "content": {
                "code": code,
                "message": message + ":" + str(exception) if exception is not None else message
            }
        }
        self.logger.error(json.dumps(msg, ensure_ascii=False))
        
    def debug(self, code, message, trace_id):
        msg =  {
            "level": "DEBUG",
            "time": str(datetime.datetime.now()),
            "trace_id": trace_id,
            "type": LOG_TYPE,
            "project": PROJECT_NAME,
            "content": {
                "code": code,
                "message": message
            }
        }
        self.logger.debug(json.dumps(msg, ensure_ascii=False))
        
    def warning(self, code, message, trace_id):
        msg =  {
            "level": "WARNING",
            "time": str(datetime.datetime.now()),
            "trace_id": trace_id,
            "type": LOG_TYPE,
            "project": PROJECT_NAME,
            "content": {
                "code": code,
                "message": message
            }
        }
        self.logger.warning(json.dumps(msg, ensure_ascii=False))

    def log(self, level, trace_id, content):
        if not self.debug:
            message = {
                "level": self.logging_collection[level],
                "time": str(datetime.datetime.now()),
                "trace_id": trace_id,
                "type": LOG_TYPE,
                "project": PROJECT_NAME,
                "content": content
            }
        else:
            message = content
        self.func_collection[level](json.dumps(message, ensure_ascii=False))


if __name__ == '__main__':
    logger = Log()      
    # logger.log(logging.INFO, '222', {1: '222'})
    logger.error(20000, "success", "313241234")
