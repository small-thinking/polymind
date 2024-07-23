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
        # Reset the Logger singleton for each test
        Logger._instance = None
        Logger._initialized = False

        log_folder = tmp_path / "logs"
        log_folder.mkdir()
        logger = Logger(logger_name="test_logger", display_level="DEBUG")
        return logger

    def test_initialization(self, setup_logger):
        logger = setup_logger
        assert logger.logging_level == Logger.LoggingLevel.DEBUG
        assert logger.logger.level == Logger.LoggingLevel.DEBUG.value
        assert len(logger.logger.handlers) == 1
        assert logger._initialized == True

    def test_single_initialization(self):
        logger1 = Logger(logger_name="test_logger1", display_level="DEBUG")
        logger2 = Logger(logger_name="test_logger2", display_level="INFO")

        assert logger1 is logger2, "Logger is not implementing the singleton pattern correctly"
        assert (
            logger1.logging_level == Logger.LoggingLevel.DEBUG
        ), "Logger level should not change after first initialization"
        assert len(logger1.logger.handlers) == 1, "Only one handler should be added"

    @pytest.mark.parametrize(
        "log_method, log_level, color, expected_log_method",
        [
            ("debug", Logger.LoggingLevel.DEBUG, Fore.BLACK, "debug"),
            ("info", Logger.LoggingLevel.INFO, Fore.WHITE, "info"),
            ("tool_log", Logger.LoggingLevel.TOOL, Fore.YELLOW, "log"),
            ("task_log", Logger.LoggingLevel.TASK, Fore.BLUE, "log"),
            ("thought_process_log", Logger.LoggingLevel.THOUGHT_PROCESS, Fore.GREEN, "log"),
            ("warning", Logger.LoggingLevel.WARNING, Fore.YELLOW, "warning"),
            ("error", Logger.LoggingLevel.ERROR, Fore.RED, "error"),
            ("critical", Logger.LoggingLevel.CRITICAL, Fore.MAGENTA, "critical"),
        ],
    )
    @patch("inspect.stack", return_value=[None, None, None, MagicMock(function="test_func", lineno=42)])
    def test_log_methods(self, mock_stack, setup_logger, log_method, log_level, color, expected_log_method):
        logger = setup_logger
        log_func = getattr(logger, log_method)

        with patch.object(logger.logger, expected_log_method) as mock_log_method:
            log_func("Test message")

            log_message = "test_func(42): Test message"
            expected_log_message = f"{color}{log_message}{Fore.RESET}"

            if expected_log_method == "log":
                mock_log_method.assert_called_once_with(log_level.value, expected_log_message)
            else:
                mock_log_method.assert_called_once_with(expected_log_message)

    def test_logging_levels_order(self):
        assert (
            Logger.LoggingLevel.DEBUG.value
            < Logger.LoggingLevel.INFO.value
            < Logger.LoggingLevel.TOOL.value
            < Logger.LoggingLevel.TASK.value
            < Logger.LoggingLevel.THOUGHT_PROCESS.value
            < Logger.LoggingLevel.WARNING.value
            < Logger.LoggingLevel.ERROR.value
            < Logger.LoggingLevel.CRITICAL.value
        ), "Logging levels are not in the correct order"

    def test_custom_logging_levels(self):
        assert Logger.LoggingLevel.TOOL.value == 25, "TOOL logging level is incorrect"
        assert Logger.LoggingLevel.TASK.value == 26, "TASK logging level is incorrect"
        assert Logger.LoggingLevel.THOUGHT_PROCESS.value == 27, "THOUGHT_PROCESS logging level is incorrect"

    def test_from_string_method(self):
        assert Logger.LoggingLevel.from_string("DEBUG") == Logger.LoggingLevel.DEBUG
        assert Logger.LoggingLevel.from_string("info") == Logger.LoggingLevel.INFO
        assert Logger.LoggingLevel.from_string("CRITICAL") == Logger.LoggingLevel.CRITICAL

        with pytest.raises(ValueError):
            Logger.LoggingLevel.from_string("INVALID_LEVEL")

    @patch("logging.getLogger")
    @patch("logging.StreamHandler")
    def test_custom_log_levels_added(self, mock_stream_handler, mock_get_logger):
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        mock_handler = MagicMock()
        mock_stream_handler.return_value = mock_handler

        Logger(logger_name="test_logger")

        assert logging.getLevelName(25) == "TOOL", "TOOL log level not added"
        assert logging.getLevelName(26) == "TASK", "TASK log level not added"
        assert logging.getLevelName(27) == "THOUGHT_PROCESS", "THOUGHT_PROCESS log level not added"
