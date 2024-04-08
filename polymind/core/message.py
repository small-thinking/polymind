import json
from typing import Any, Dict, List

from pydantic import BaseModel, field_validator


class Message(BaseModel):
    """Message is a class that represents a message that can carry any information.
    There are a reserved input_fields.
    They are with type List[Dict[str, str]]. Each Dict[str, str] stores the properties
    of a field.
    An example is:

    fields = [
        {"name": "input", "type": "str", "required": True},
        {"name": "max_tokens", "type": "int", "required": False}
    ]
    """

    content: Dict[str, Any]

    @field_validator("content")
    def check_content(cls, value):
        """Check if the content is a dictionary."""
        if not isinstance(value, dict):
            raise ValueError("Content must be a dictionary")
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self.content.get(key, default)

    def get_fields(self) -> List[Dict[str, str]]:
        return self.content.get("fields", {})

    def get_fields_json(self) -> str:
        return json.dumps(self.get_fields())

    def set(self, key: str, value: Any):
        if "fields" not in self.content:
            self.content["fields"] = {}
        # Overwrite the field if it already exists
        self.content["fields"][key] = {
            "name": key,
            "type": type(value).__name__,
            # By default, all fields are required
        }
        self.content[key] = value
