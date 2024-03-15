"""Test cases for OpenAIChatTool.
Run the test with the following command:
    poetry run pytest tests/polymind/tools/test_oai_tools.py
"""

"""Test cases for OpenAIChatTool."""

import pytest
from unittest.mock import AsyncMock, patch
from polymind.core.message import Message
from polymind.tools.oai_tools import OpenAIChatTool


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Fixture to mock environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")


@pytest.fixture
def tool(mock_env_vars):
    """Fixture to create an instance of OpenAIChatTool with mocked environment variables."""
    llm_name = "gpt-4-turbo"
    system_prompt = "You are an orchestrator"
    return OpenAIChatTool(llm_name=llm_name, system_prompt=system_prompt)


@pytest.mark.asyncio
async def test_execute_success(tool):
    """Test _execute method of OpenAIChatTool for successful API call."""
    prompt = "How are you?"
    system_prompt = "You are a helpful AI assistant."
    expected_response_content = "I'm doing great, thanks for asking!"

    # Patch the specific instance of AsyncOpenAI used by our tool instance
    with patch.object(
        tool.client.chat.completions, "create", new_callable=AsyncMock
    ) as mock_create:
        mock_create.return_value = AsyncMock(
            choices=[AsyncMock(message=AsyncMock(content=expected_response_content))]
        )
        input_message = Message(
            content={"prompt": prompt, "system_prompt": system_prompt}
        )
        response_message = await tool._execute(input_message)

    assert response_message.content["response"] == expected_response_content
    mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_execute_failure_empty_prompt(tool):
    """Test _execute method of OpenAIChatTool raises ValueError with empty prompt."""
    with pytest.raises(ValueError) as excinfo:
        await tool._execute(Message(content={"prompt": ""}))
    assert "Prompt cannot be empty." in str(excinfo.value)
