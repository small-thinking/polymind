"""
Run with the command:
    poetry run pytest tests/polymind/core_tools/test_index_tool.py
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from polymind.core.message import Message
from polymind.core_tools.index_tool import (  # Replace 'your_module' with the actual module name
    KnowledgeIndexTool, ToolIndexer)


@pytest.mark.asyncio
class TestKnowledgeIndexTool:

    @pytest.fixture
    def mock_milvus_client(self):
        with patch("pymilvus.MilvusClient") as mock:
            instance = mock.return_value
            instance.create_collection = AsyncMock()
            instance.drop_collection = AsyncMock()
            instance.insert = AsyncMock()
            yield instance

    @pytest.fixture(autouse=True)
    def tool(self, mock_milvus_client):
        tool = KnowledgeIndexTool()
        tool._client = mock_milvus_client
        tool.embedder = AsyncMock()
        return tool

    async def test_index_empty_items(self, tool):
        """Test the case when the items to index is empty."""
        with pytest.raises(ValueError, match="The items should not be empty."):
            input_message = Message(content={"items": [], "key_to_index": "title"})
            await tool(input_message)

    async def test_index_invalid_key_to_index(self, tool):
        """Test the case when the key to index is not in the items."""
        with pytest.raises(ValueError, match="The key to index should not be empty."):
            input_message = Message(content={"items": [{"title": "Example", "content": "Data"}], "key_to_index": ""})
            await tool(input_message)

    async def test_index_success(self, tool):
        """Test the case when the indexing is successful."""
        input_message = Message(
            content={
                "items": [{"title": "Example", "content": "Data"}],
                "key_to_index": "title",
                "collection_name": "knowledge",
            }
        )
        response = await tool(input_message)
        tool._client.insert.assert_called_once()
        assert response.content == {"status": "success"}
