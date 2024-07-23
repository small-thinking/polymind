import inspect
import logging
import os
from enum import Enum
from typing import Optional, Union

from colorama import Fore
from dotenv import load_dotenv


class Logger:
    _instance = None

    class LoggingLevel(Enum):
        DEBUG = logging.DEBUG
        INFO = logging.INFO
        TOOL = 25
        TASK = 26
        THOUGHT_PROCESS = 27
        WARNING = logging.WARNING
        ERROR = logging.ERROR
        CRITICAL = logging.CRITICAL

        @classmethod
        def from_string(cls, level_string: str):
            try:
                return cls[level_string.upper()]
            except KeyError:
                raise ValueError(f"Invalid logging level: {level_string}")

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        logger_name: str,
        verbose: bool = True,
        display_level: Optional[Union[LoggingLevel, str]] = None,
    ):
        load_dotenv(override=True)

        if display_level is None:
            env_level = os.getenv("LOGGING_LEVEL", "INFO")
            self.logging_level = self.LoggingLevel.from_string(env_level)
        elif isinstance(display_level, str):
            self.logging_level = self.LoggingLevel.from_string(display_level)
        else:
            self.logging_level = display_level

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(self.logging_level.value)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s (%(filename)s:%(lineno)d)")
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(self.logging_level.value)
        self.console_handler.setFormatter(self.formatter)
        self.logger.addHandler(self.console_handler)

        # Add custom log levels
        logging.addLevelName(self.LoggingLevel.TOOL.value, "TOOL")
        logging.addLevelName(self.LoggingLevel.TASK.value, "TASK")
        logging.addLevelName(self.LoggingLevel.THOUGHT_PROCESS.value, "THOUGHT_PROCESS")

    def _log(self, message: str, level: LoggingLevel, color: str) -> None:
        if level.value >= self.logging_level.value:
            if len(inspect.stack()) >= 4:
                caller_frame = inspect.stack()[3]
            else:
                caller_frame = inspect.stack()[2]
            caller_name = caller_frame.function
            caller_line = caller_frame.lineno
            message = f"{caller_name}({caller_line}): {message}"
            log_message = color + message + Fore.RESET

            if level == self.LoggingLevel.DEBUG:
                self.logger.debug(log_message)
            elif level == self.LoggingLevel.INFO:
                self.logger.info(log_message)
            elif level == self.LoggingLevel.TOOL:
                self.logger.log(level.value, log_message)
            elif level == self.LoggingLevel.TASK:
                self.logger.log(level.value, log_message)
            elif level == self.LoggingLevel.THOUGHT_PROCESS:
                self.logger.log(level.value, log_message)
            elif level == self.LoggingLevel.WARNING:
                self.logger.warning(log_message)
            elif level == self.LoggingLevel.ERROR:
                self.logger.error(log_message)
            elif level == self.LoggingLevel.CRITICAL:
                self.logger.critical(log_message)

    def debug(self, message: str) -> None:
        self._log(message, self.LoggingLevel.DEBUG, Fore.BLACK)

    def info(self, message: str) -> None:
        self._log(message, self.LoggingLevel.INFO, Fore.WHITE)

    def tool_log(self, message: str) -> None:
        self._log(message, self.LoggingLevel.TOOL, Fore.YELLOW)

    def task_log(self, message: str) -> None:
        self._log(message, self.LoggingLevel.TASK, Fore.BLUE)

    def thought_process_log(self, message: str) -> None:
        self._log(message, self.LoggingLevel.THOUGHT_PROCESS, Fore.GREEN)

    def warning(self, message: str) -> None:
        self._log(message, self.LoggingLevel.WARNING, Fore.YELLOW)

    def error(self, message: str) -> None:
        self._log(message, self.LoggingLevel.ERROR, Fore.RED)

    def critical(self, message: str) -> None:
        self._log(message, self.LoggingLevel.CRITICAL, Fore.MAGENTA)
