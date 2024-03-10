"""Run the test with the following command:
    poetry run pytest tests/polymind/core/test_message.py
"""

from polymind.core.message import Message


class TestMessage:
    def test_message_creation(self):
        content = {"key": "value"}
        message = Message(content=content)
        assert message.content == content

    def test_message_get_set(self):
        message = Message(content={})
        key, value = "test_key", "test_value"
        message.set(key, value)
        assert message.get(key) == value
