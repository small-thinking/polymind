from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel, Field


class Memory(BaseModel, ABC):

    @abstractmethod
    def set_memory(self, **kwargs) -> None:
        pass

    @abstractmethod
    def get_memory(**kwargs) -> str:
        pass


class LinearMemory(Memory):
    """LinearMemory is a simple memory that stores the memory in a list.
    When retrieving the memory, it returns the last k pieces of memory.
    """

    memory_list: List[str] = Field(default=[], description="The list of memory stored in the memory.")

    def set_memory(self, piece: str, **kwargs) -> None:
        """Set the memory.

        Args:
            piece (str): The piece of memory to store.
        """
        self.memory_list.append(piece)

    def get_memory(self, last_k: int = 0, **kwargs) -> str:
        """Get the last k pieces of memory."""
        if last_k == 0 or last_k > len(self.memory_list):
            return "\n".join(self.memory_list)
        else:
            return "\n".join(self.memory_list[-last_k:])
