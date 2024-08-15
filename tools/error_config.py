
from enum import Enum


class ErrorCode(Enum):
    SUCCESS = 200
    INVALID_INPUT = 4001
    NOT_FOUND = 4002
    INTERNAL_ERROR = 4003
    PARAMERTER_ERROR = 4004