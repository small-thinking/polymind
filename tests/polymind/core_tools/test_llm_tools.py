"""Tests for OpenAIChatTool.
Run the test with the following command:
    poetry run pytest tests/polymind/core_tools/test_llm_tools.py
"""

import json
import re
import textwrap
from unittest.mock import AsyncMock, patch

import pytest

from polymind.core.message import Message
from polymind.core_tools.llm_tool import (OpenAIChatTool,
                                          OpenAICodeGenerationTool)


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
            input_message = Message(content={"input": prompt, "system_prompt": system_prompt})
            response_message = await tool(input_message)

        assert response_message.content["output"] == expected_response_content
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_failure_empty_prompt(self, tool):
        """Test _execute method of OpenAIChatTool raises ValueError with empty prompt."""
        with pytest.raises(ValueError) as excinfo:
            await tool._execute(Message(content={"input": ""}))
        assert "Prompt in the field 'input' cannot be empty." in str(excinfo.value)

    def test_get_spec(self, tool):
        """Test get_spec method of OpenAIChatTool."""
        spec_str = tool.get_spec()
        expected_json_str = """{
        "input_message": [
            {
                "name": "system_prompt",
                "type": "str",
                "required": false,
                "description": "The system prompt for the chat.",
                "example": "You are a helpful AI assistant."
            },
            {
                "name": "input",
                "type": "str",
                "required": true,
                "description": "The prompt for the chat.",
                "example": "hello, how are you?"
            },
            {
                "name": "max_tokens",
                "type": "int",
                "required": false,
                "description": "The maximum number of tokens for the chat.",
                "example": "1500"
            },
            {
                "name": "temperature",
                "type": "float",
                "required": false,
                "description": "The temperature for the chat.",
                "example": "0.7"
            },
            {
                "name": "top_p",
                "type": "float",
                "required": false,
                "description": "The top p for the chat.",
                "example": "0.1"
            }
        ],
        "output_message": [
            {
                "name": "output",
                "type": "str",
                "required": true,
                "description": "The response from the chat.",
                "example": "I'm good, how are you?"
            }
        ]
        }"""
        assert json.loads(spec_str) == json.loads(
            expected_json_str
        ), "The spec string should match the expected JSON string"


# class TestOpenAIEmbeddingTool:
#     @pytest.fixture
#     def openai_embedding_tool(self, monkeypatch):
#         monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-api-key")
#         return OpenAIEmbeddingTool(embedding_restful_tool=RestAPITool())

#     @pytest.mark.asyncio
#     async def test_embedding_success(self, openai_embedding_tool):
#         with aioresponses() as m:
#             m.post(
#                 openai_embedding_tool.url,
#                 payload={
#                     "data": [
#                         {"embedding": [0.1, 0.2, 0.3]},
#                         {"embedding": [0.4, 0.5, 0.6]},
#                     ]
#                 },
#                 status=200,
#             )

#             input_texts = ["hello, how are you?", "This is a test."]
#             embeddings = await openai_embedding_tool._embedding(input_texts)
#             assert isinstance(embeddings, np.ndarray), "Expected output to be a numpy array"
#             assert embeddings.shape == (
#                 2,
#                 3,
#             ), "Expected shape of embeddings to match input texts and dimensionality"

#     @pytest.mark.asyncio
#     async def test_embedding_failure(self, openai_embedding_tool):
#         with aioresponses() as m:
#             m.post(openai_embedding_tool.url, payload={"error": "Bad request"}, status=400)

#             input_texts = ["hello, how are you?", "This is a test."]
#             with pytest.raises(Exception) as excinfo:
#                 await openai_embedding_tool._embedding(input_texts)
#             assert "Bad request" in str(excinfo.value), "Expected failure when API returns error"


class TestOpenAICodeGenerationTool:
    @pytest.fixture
    def llm_tool_mock(self):
        return AsyncMock()

    @pytest.fixture
    def code_gen_tool(self, llm_tool_mock):
        codegen_tool = OpenAICodeGenerationTool(max_attempts=1)
        codegen_tool._llm_tool = llm_tool_mock
        return codegen_tool

    @pytest.mark.asyncio
    async def test_execute_successful(self, code_gen_tool, llm_tool_mock):
        # Setup mock responses with different outcomes for each call
        llm_tool_mock.side_effect = [
            AsyncMock(  # Called by code gen
                content={
                    "output": """
                    ```python
                    import yfinance as yf
                    import json
                    output = {"result": 42}
                    print(json.dumps(output))
                    ```
                    """
                }
            ),
            AsyncMock(  # Called by output parse
                content={
                    "output": """
                    ```json
                    {
                        "status": "success",
                        "output": {"result": 42}
                    }
                    ```
                    """
                }
            ),
        ]

        # Define expected outcomes
        expected_code = textwrap.dedent(
            """
            import yfinance as yf
            import json
            output = {"result": 42}
            print(json.dumps(output))
        """
        ).strip()
        expected_output = '{"result": 42}'

        input_message = Message(content={"query": "Sum two numbers"})

        result = await code_gen_tool(input_message)
        actual_output = result.content["output"]

        assert textwrap.dedent(result.content["code"]).strip() == expected_code
        assert json.loads(actual_output) == json.loads(expected_output)

    @pytest.mark.asyncio
    async def test_execute_failure_max_attempts(self, code_gen_tool, llm_tool_mock):
        # Simulate failures
        llm_tool_mock.side_effect = [AsyncMock(side_effect=Exception("Error")) for _ in range(3)]
        input_message = Message(content={"input": "Sum two numbers"})

        with pytest.raises(Exception):
            await code_gen_tool(input_message)

    @pytest.mark.asyncio
    async def test_code_gen_success(self, code_gen_tool, llm_tool_mock):
        # Setup mock response
        llm_tool_mock.return_value = AsyncMock(
            content={
                "output": """
                ```python
                a = 10
                b = 32
                def total(a, b):
                    return a + b
                output = {"result": total(a, b)}
                ```
                """
            }
        )
        requirement = "Sum two numbers"
        previous_errors = []

        code = await code_gen_tool._code_gen(requirement=requirement, previous_errors=previous_errors)
        expected_code = """
            a = 10
            b = 32
            def total(a, b):
                return a + b
            output = {"result": total(a, b)}
        """
        expected_code = re.sub("\s+", "", expected_code)
        assert re.sub("\s+", "", code) == expected_code

    @pytest.mark.asyncio
    async def test_code_gen_failure_no_code(self, code_gen_tool, llm_tool_mock):
        # Setup mock response
        llm_tool_mock.return_value = AsyncMock(content={"output": ""})
        requirement = "Sum two numbers"
        previous_errors = []

        with pytest.raises(ValueError):
            await code_gen_tool._code_gen(requirement, previous_errors)

    def test_extract_required_packages(self, code_gen_tool):
        code = textwrap.dedent(
            """
            import numpy
            import pandas as pd
            import matplotlib.pyplot as plt
            import yfinance as yf
            from sklearn.linear_model import LinearRegression
            """
        )
        expected_packages = set(["numpy", "pandas", "matplotlib", "yfinance", "sklearn"])
        packages = code_gen_tool._extract_required_packages(code)
        assert set(packages) == expected_packages

    @pytest.mark.asyncio
    async def test_code_run_valid_python(self, code_gen_tool):
        code = textwrap.dedent(
            """
            import json
            output = {"result": 100, "price": 200, "name": "test", "ids": [1, 2, 3]}
            print(json.dumps(output))
        """
        ).strip()
        result = await code_gen_tool._code_run(code)
        assert json.loads(result) == {"result": 100, "price": 200, "name": "test", "ids": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_code_run_invalid_python(self, code_gen_tool):
        code = "for i in range(10 print(i)"
        with pytest.raises(Exception):
            await code_gen_tool._code_run(code)

    @pytest.mark.asyncio
    async def test_output_parse_success(self, code_gen_tool):
        # Expected output from the subprocess
        mock_stdout = """
        {
            "status": "success",
            "output": {
                "result": 42
            }
        }
        """

        # Mock subprocess execution to return success and custom stdout
        with patch("asyncio.create_subprocess_exec") as mock_subproc_exec:
            # Setup mock process with desired behavior
            mock_proc = AsyncMock()
            mock_proc.communicate.return_value = (mock_stdout.encode(), b"")  # no stderr output
            mock_proc.returncode = 0  # Simulate successful execution
            mock_subproc_exec.return_value = mock_proc

            requirement = "Sum two numbers"
            code_gen_output = """
            {
                "result": 42
            }
            """
            expected_output = {"status": "success", "output": {"result": 42}}

            # This method seems to parse the output; ensure it works with mocked stdout
            parsed_output = await code_gen_tool._code_run(code_gen_output)
            assert json.loads(parsed_output) == expected_output

    @pytest.mark.asyncio
    async def test_output_parse_success(self, code_gen_tool, llm_tool_mock):
        # Setup mock response
        llm_tool_mock.return_value = AsyncMock(
            content={
                "output": """
                ```json
                {
                    "status": "success",
                    "output": {
                        "result": 42
                    }
                }
                ```
                """
            }
        )
        requirement = "Sum two numbers"
        code_gen_output = """
        {
            "result": 42
        }
        """
        expected_output = {"result": 42}

        parsed_output = await code_gen_tool._output_parse(requirement=requirement, output=code_gen_output)
        assert json.loads(parsed_output) == expected_output

    @pytest.mark.asyncio
    async def test_output_parse_failure(self, code_gen_tool, llm_tool_mock):
        # Setup mock response
        llm_tool_mock.return_value = AsyncMock(
            content={"output": json.dumps({"status": "failure", "reason": "Error: Invalid input"})}
        )
        requirement = "Sum two numbers"
        output = {"result": 42}

        with pytest.raises(ValueError):
            await code_gen_tool._output_parse(requirement=requirement, output=output)
