"""
Run command to test the code:
    poetry run pytest tests/polymind/tools/test_tool_management_tool.py
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, patch

import faiss
import numpy as np
import pytest

from polymind.core.message import Message
from polymind.tools.llm_tool import OpenAIEmbeddingTool
from polymind.tools.tool_management_tool import ToolIndexer, ToolRetriever


class TestToolIndexerIntegration:
    @pytest.fixture
    def tool_indexer(self):
        temp_dir = tempfile.TemporaryDirectory()
        indexer = ToolIndexer(learned_tool_folder=temp_dir.name)
        yield indexer
        temp_dir.cleanup()

    @pytest.fixture
    def input_message(self):
        return Message(
            content={
                "tool_name": "test_tool",
                "descriptions": [
                    "This tool does something amazing.",
                    "It helps with testing.",
                    "The third description for good measure.",
                ],
                "tool_file_name": "test_tool.py",
            }
        )

    @pytest.fixture
    def embeddings(self):
        return np.random.rand(3, 128).astype("float32")

    @pytest.mark.asyncio
    async def test_index_creation_and_query(self, tool_indexer, input_message, embeddings):
        # Use patch to mock the _embedding method
        tool_indexer.embedder._embedding = AsyncMock(return_value=embeddings)
        await tool_indexer._execute(input_message)

        index_path = os.path.join(tool_indexer.learned_tool_folder, "tool.index")
        metadata_path = os.path.join(tool_indexer.learned_tool_folder, "tool_profiles.json")
        assert os.path.exists(index_path), "Index file should exist after execution"
        assert os.path.exists(metadata_path), "Metadata file should exist after execution"
        # Find top 2
        index = faiss.read_index(index_path)
        distances, indices = index.search(embeddings, k=2)
        assert indices.shape == (3, 2), "Should return correct number of indices"
        assert distances.shape == (
            3,
            2,
        ), "Should return correct number of distances"

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        assert len(metadata) == 3, "There should be one tool indexed 3 times"
        tool_metadata = metadata[0]
        assert tool_metadata["tool_name"] == input_message.content["tool_name"], "Tool name should match"
        assert tool_metadata["file_name"] == input_message.content["tool_file_name"], "File name should match"
        assert tool_metadata["descriptions"] == input_message.content["descriptions"], "Descriptions should match"


class TestToolRetriever:
    @pytest.fixture
    def tool_retriever(self):
        # Specify the path to your test folder that contains tool_profiles.json and tool.index
        test_folder_path = "./tests/test_data/tool_index"
        # Initialize ToolRetriever with the test folder path
        return ToolRetriever(learned_tool_folder=test_folder_path)

    @pytest.mark.asyncio
    async def test_retrieval(self, tool_retriever):
        requirement = "I need a tool for testing."
        input_message = Message(content={"requirement": requirement})

        # Mock the embedding operation to return the structured JSON response you described
        mock_embedding_response = np.random.rand(1, 1536).astype("float32")
        mock_embedding = AsyncMock(return_value=mock_embedding_response)
        with patch.object(OpenAIEmbeddingTool, "_embedding", mock_embedding):
            output_message = await tool_retriever(input_message)

        # Verify the output
        assert "candidates" in output_message.content, "Output message should contain candidates"
        candidates = output_message.content["candidates"]
        assert len(candidates) > 0, "There should be at least one candidate returned"
        for candidate in candidates:
            assert (
                "tool_name" in candidate and "descriptions" in candidate
            ), "Each candidate should contain 'tool_name' and 'descriptions'"
