"""Run the test with the following command:
    poetry run pytest tests/polymind/core/test_message.py
"""

import json

import pytest

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

    def test_message_content_not_dict(self):
        with pytest.raises(ValueError) as excinfo:
            Message(content="invalid")
            assert False, "Message should not be created"

    def test_get_fields(self):
        message = Message(content={"fields": {"key": {"name": "key", "type": "str"}}})
        assert message.get_fields() == {"key": {"name": "key", "type": "str"}}
        # Simulate set the fields
        message = Message(content={})
        message.set("name", "name1")
        message.set("age", 23)
        message.set("items_ids", [1, 2, 3])
        assert message.get_fields() == {
            "name": {"name": "name", "type": "str"},
            "age": {"name": "age", "type": "int"},
            "items_ids": {"name": "items_ids", "type": "list"},
        }
        assert message.get_fields_json() == json.dumps(
            {
                "name": {"name": "name", "type": "str"},
                "age": {"name": "age", "type": "int"},
                "items_ids": {"name": "items_ids", "type": "list"},
            }
        )
