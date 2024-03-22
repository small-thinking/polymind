"""Tests for OpenAIChatTool.
Run the test with the following command:
    poetry run pytest tests/polymind/tools/test_llm_tools.py
"""

import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
from aioresponses import aioresponses

from polymind.core.message import Message
from polymind.tools.llm_tool import OpenAIChatTool, OpenAIEmbeddingTool
from polymind.tools.rest_api_tool import RestAPITool


class TestOpenAIChatTool:
    @pytest.fixture
    def mock_env_vars(self, monkeypatch):
        """Fixture to mock environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "test_key")

    @pytest.fixture
    def tool(self, mock_env_vars):
        """Fixture to create an instance of OpenAIChatTool with mocked environment variables."""
        llm_name = "gpt-4-turbo"
        system_prompt = "You are an orchestrator"
        return OpenAIChatTool(llm_name=llm_name, system_prompt=system_prompt)

    @pytest.mark.asyncio
    async def test_execute_success(self, tool):
        """Test _execute method of OpenAIChatTool for successful API call."""
        prompt = "How are you?"
        system_prompt = "You are a helpful AI assistant."
        expected_response_content = "I'm doing great, thanks for asking!"

        # Patch the specific instance of AsyncOpenAI used by our tool instance
        with patch.object(tool.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = AsyncMock(
                choices=[AsyncMock(message=AsyncMock(content=expected_response_content))]
            )
            input_message = Message(content={"prompt": prompt, "system_prompt": system_prompt})
            response_message = await tool._execute(input_message)

        assert response_message.content["response"] == expected_response_content
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_failure_empty_prompt(self, tool):
        """Test _execute method of OpenAIChatTool raises ValueError with empty prompt."""
        with pytest.raises(ValueError) as excinfo:
            await tool._execute(Message(content={"prompt": ""}))
        assert "Prompt cannot be empty." in str(excinfo.value)

    def test_get_spec(self, tool):
        """Test get_spec method of OpenAIChatTool."""
        spec_str = tool.get_spec()
        expected_json_str = """{
        "input_message": [
            {
                "name": "system_prompt",
                "type": "str",
                "description": "The system prompt for the chat.",
                "example": "You are a helpful AI assistant."
            },
            {
                "name": "prompt",
                "type": "str",
                "description": "The prompt for the chat.",
                "example": "hello, how are you?"
            },
            {
                "name": "max_tokens",
                "type": "int",
                "description": "The maximum number of tokens for the chat.",
                "example": "1500"
            },
            {
                "name": "temperature",
                "type": "float",
                "description": "The temperature for the chat.",
                "example": "0.7"
            },
            {
                "name": "top_p",
                "type": "float",
                "description": "The top p for the chat.",
                "example": "0.1"
            }
        ],
        "output_message": [
            {
                "name": "response",
                "type": "str",
                "description": "The response from the chat.",
                "example": "I'm good, how are you?"
            }
        ]
        }"""
        assert json.loads(spec_str) == json.loads(
            expected_json_str
        ), "The spec string should match the expected JSON string"


class TestOpenAIEmbeddingTool:
    @pytest.fixture
    def openai_embedding_tool(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-api-key")
        return OpenAIEmbeddingTool(embedding_restful_tool=RestAPITool())

    @pytest.mark.asyncio
    async def test_embedding_success(self, openai_embedding_tool):
        with aioresponses() as m:
            m.post(
                openai_embedding_tool.url,
                payload={
                    "data": [
                        {"embedding": [0.1, 0.2, 0.3]},
                        {"embedding": [0.4, 0.5, 0.6]},
                    ]
                },
                status=200,
            )

            input_texts = ["hello, how are you?", "This is a test."]
            embeddings = await openai_embedding_tool._embedding(input_texts)
            assert isinstance(embeddings, np.ndarray), "Expected output to be a numpy array"
            assert embeddings.shape == (
                2,
                3,
            ), "Expected shape of embeddings to match input texts and dimensionality"

    @pytest.mark.asyncio
    async def test_embedding_failure(self, openai_embedding_tool):
        with aioresponses() as m:
            m.post(openai_embedding_tool.url, payload={"error": "Bad request"}, status=400)

            input_texts = ["hello, how are you?", "This is a test."]
            with pytest.raises(Exception) as excinfo:
                await openai_embedding_tool._embedding(input_texts)
            assert "Bad request" in str(excinfo.value), "Expected failure when API returns error"
