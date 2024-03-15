from typing import Any, Dict

from pydantic import BaseModel, field_validator


class Message(BaseModel):
    """Message is a class that represents a message that can carry any information."""

    content: Dict[str, Any]

    @field_validator("content")
    def check_content(cls, value):
        """Check if the content is a dictionary."""
        if not isinstance(value, dict):
            raise ValueError("Content must be a dictionary")
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self.content.get(key, default)

    def set(self, key: str, value: Any):
        self.content[key] = value
