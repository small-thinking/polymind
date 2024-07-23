"""Run the tests with command:
    poetry run pytest tests/polymind/core/test_logger.py -vv
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from colorama import Fore

from polymind.core.logger import Logger


class TestLogger:

    @pytest.fixture(scope="function")
    def setup_logger(self, tmp_path):
        log_folder = tmp_path / "logs"
        log_folder.mkdir()
        logger = Logger(logger_name="test_logger", display_level="INFO")
        return logger

    def test_initialization(self, setup_logger):
        logger = setup_logger
        assert (
            logger.logging_level == Logger.LoggingLevel.INFO
        ), f"Logger level is incorrect: {logger.logging_level} != INFO"
        assert logger.logger.level == logging.INFO, f"Logger level is not set correctly: {logger.logger.level} != INFO"
        assert len(logger.logger.handlers) == 1, "Logger handlers are not initialized properly"

    def test_logging_levels(self, setup_logger):
        logger = setup_logger
        logger.info("Info message")
        logger.warning("Warning message")
        assert (
            logger.logging_level == Logger.LoggingLevel.INFO
        ), f"Logger level is incorrect: {logger.logging_level} != INFO"
        assert logger.logger.level == logging.INFO, f"Logger level is not set correctly: {logger.logger.level} != INFO"

    @patch("inspect.stack", return_value=[None, None, None, MagicMock(function="test_func", lineno=42)])
    def test_log_method(self, mock_stack, setup_logger):
        logger = setup_logger
        logger.log("Test log message", Logger.LoggingLevel.INFO, color=Fore.WHITE)
        log_message = "test_func(42): Test log message"
        log_output = f"{Fore.WHITE}{log_message}{Fore.RESET}"
        formatted_message = logger.formatter.format(
            logging.LogRecord(
                name="test_logger",
                level=logging.INFO,
                pathname="",
                lineno=42,
                msg=log_output,
                args=None,
                exc_info=None,
            )
        )
        assert log_message in formatted_message, "Log message is incorrect"

    def test_debug_log(self, setup_logger):
        logger = setup_logger
        logger.debug("Debug message")
        log_message = "pytest_pyfunc_call(1): Debug message"
        log_output = f"{Fore.BLACK}{log_message}{Fore.RESET}"
        formatted_message = logger.formatter.format(
            logging.LogRecord(
                name="test_logger",
                level=logging.DEBUG,
                pathname="",
                lineno=1,
                msg=log_output,
                args=None,
                exc_info=None,
            )
        )
        assert log_message in formatted_message, "Debug log message is incorrect"

    def test_info_log(self, setup_logger):
        logger = setup_logger
        logger.info("Info message")
        log_message = "pytest_pyfunc_call(1): Info message"
        log_output = f"{Fore.WHITE}{log_message}{Fore.RESET}"
        formatted_message = logger.formatter.format(
            logging.LogRecord(
                name="test_logger",
                level=logging.INFO,
                pathname="",
                lineno=1,
                msg=log_output,
                args=None,
                exc_info=None,
            )
        )
        assert log_message in formatted_message, "Info log message is incorrect"

    def test_tool_log(self, setup_logger):
        logger = setup_logger
        logger.tool_log("Tool log message")
        log_message = "pytest_pyfunc_call(1): Tool log message"
        log_output = f"{Fore.YELLOW}{log_message}{Fore.RESET}"
        formatted_message = logger.formatter.format(
            logging.LogRecord(
                name="test_logger",
                level=logging.INFO,
                pathname="",
                lineno=1,
                msg=log_output,
                args=None,
                exc_info=None,
            )
        )
        assert log_message in formatted_message, "Tool log message is incorrect"

    def test_task_log(self, setup_logger):
        logger = setup_logger
        logger.task_log("Task log message")
        log_message = "pytest_pyfunc_call(1): Task log message"
        log_output = f"{Fore.BLUE}{log_message}{Fore.RESET}"
        formatted_message = logger.formatter.format(
            logging.LogRecord(
                name="test_logger",
                level=logging.INFO,
                pathname="",
                lineno=1,
                msg=log_output,
                args=None,
                exc_info=None,
            )
        )
        assert log_message in formatted_message, "Task log message is incorrect"

    def test_thought_process_log(self, setup_logger):
        logger = setup_logger
        logger.thought_process_log("Thought process log message")
        log_message = "pytest_pyfunc_call(1): Thought process log message"
        log_output = f"{Fore.GREEN}{log_message}{Fore.RESET}"
        formatted_message = logger.formatter.format(
            logging.LogRecord(
                name="test_logger",
                level=logging.INFO,
                pathname="",
                lineno=1,
                msg=log_output,
                args=None,
                exc_info=None,
            )
        )
        assert log_message in formatted_message, "Thought process log message is incorrect"

    def test_warning_log(self, setup_logger):
        logger = setup_logger
        logger.warning("Warning message")
        log_message = "pytest_pyfunc_call(1): Warning message"
        log_output = f"{Fore.YELLOW}{log_message}{Fore.RESET}"
        formatted_message = logger.formatter.format(
            logging.LogRecord(
                name="test_logger",
                level=logging.WARNING,
                pathname="",
                lineno=1,
                msg=log_output,
                args=None,
                exc_info=None,
            )
        )
        assert log_message in formatted_message, "Warning log message is incorrect"

    def test_error_log(self, setup_logger):
        logger = setup_logger
        logger.error("Error message")
        log_message = "pytest_pyfunc_call(1): Error message"
        log_output = f"{Fore.RED}{log_message}{Fore.RESET}"
        formatted_message = logger.formatter.format(
            logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="",
                lineno=1,
                msg=log_output,
                args=None,
                exc_info=None,
            )
        )
        assert log_message in formatted_message, "Error log message is incorrect"

    def test_critical_log(self, setup_logger):
        logger = setup_logger
        logger.critical("Critical message")
        log_message = "pytest_pyfunc_call(1): Critical message"
        log_output = f"{Fore.MAGENTA}{log_message}{Fore.RESET}"
        formatted_message = logger.formatter.format(
            logging.LogRecord(
                name="test_logger",
                level=logging.CRITICAL,
                pathname="",
                lineno=1,
                msg=log_output,
                args=None,
                exc_info=None,
            )
        )
        assert log_message in formatted_message, "Critical log message is incorrect"
