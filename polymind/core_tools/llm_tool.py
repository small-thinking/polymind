"""
This file contains the necessary tools of using OpenAI models.
For now, it contains two tools: OpenAIChatTool and OpenAIEmbeddingTool.
"""

import datetime
import os
from typing import List

import anthropic
from openai import AsyncOpenAI
from pydantic import Field

from polymind.core.codegen import CodeGenerationTool
from polymind.core.logger import Logger
from polymind.core.message import Message
from polymind.core.tool import BaseTool, Embedder, LLMTool, Param
from polymind.core_tools.rest_api_tool import RestAPITool


class OpenAIChatTool(LLMTool):
    """OpenAITool is a bridge to OpenAI APIs.
    The tool can be initialized with llm_name, system_prompt, max_tokens, and temperature.
    The input message of this tool should contain a "prompt", and optionally a "system_prompt".
    The "system_prompt" in the input message will override the default system_prompt.
    The tool will return a message with key "answer" with the response from the OpenAI chat API.
    """

    model_config = {
        "arbitrary_types_allowed": True,  # Allow arbitrary types
    }
    tool_name: str = "open-ai-chat"
    descriptions: List[str] = [
        "This tool is used to chat with OpenAI's language models.",
        "This tool can be used as the orchestrator to control the conversation and problem solving.",
        "This tool can be used to breakdown the problem into smaller parts and solve them.",
        "This tool can be used to generate the response from the chat.",
        "This tool can be used to generate the code of new tools.",
        "This tool can do simple calculation.",
        "Simple calculator that does basic arithmetic calculation.",
    ]
    client: AsyncOpenAI = Field(default=None)
    # llm_name: str = Field(default="gpt-4o-2024-05-13")
    llm_name: str = Field(default="gpt-3.5-turbo")
    system_prompt: str = Field(
        default="You are a helpful AI assistant. You need to communicate with the user in their language."
    )
    max_tokens: int = Field(default=1500)
    temperature: float = Field(default=0.7)
    stop: str = Field(default=None)
    response_format: str = Field(default="text", description="The format of the response from the chat.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger = Logger(__file__)

    def _set_client(self):
        """Set the client for the language model."""
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def input_spec(self) -> List[Param]:
        """Return the input specification of the tool.
        The derived class must implement this method to define the input specification of the tool.

        Returns:
            List[Param]: The input specification of the tool.
        """
        return [
            Param(
                name="system_prompt",
                type="str",
                required=False,
                example="You are a helpful AI assistant.",
                description="The system prompt for the chat.",
            ),
            Param(
                name="input",
                type="str",
                required=True,
                example="hello, how are you?",
                description="The prompt for the chat.",
            ),
            Param(
                name="max_tokens",
                type="int",
                required=False,
                example="1500",
                description="The maximum number of tokens for the chat.",
            ),
            Param(
                name="temperature",
                type="float",
                required=False,
                example="0.7",
                description="The temperature for the chat.",
            ),
            Param(
                name="top_p",
                type="float",
                required=False,
                example="0.1",
                description="The top p for the chat.",
            ),
        ]

    def output_spec(self) -> List[Param]:
        """Return the output specification of the tool.
        The derived class must implement this method to define the output specification of the tool.

        Returns:
            List[Param]: The output specification of the tool.
        """
        return [
            Param(
                name="output",
                type="str",
                required=True,
                example="I'm good, how are you?",
                description="The response from the chat.",
            ),
        ]

    async def _invoke(self, input: Message) -> Message:
        """Execute the tool and return the result.
        The derived class must implement this method to define the behavior of the tool.

        Args:
            input (Message): The input to the tool carried in a message. The message should contain the below keys:
                - prompt: The prompt for the chat.
                - system_prompt: The system prompt for the chat.
                - max_tokens: The maximum number of tokens for the chat.
                - temperature: The temperature for the chat.
                - top_p: The top p for the chat.
                - stop: The stop sequence for the chat.

        Returns:
            Message: The result of the tool carried in a message.
        """
        prompt = input.get("input", "")
        temperature = input.get("temperature", self.temperature)
        max_tokens = input.get("max_tokens", self.max_tokens)
        top_p = input.get("top_p", self.top_p)
        stop = input.get("stop", self.stop)
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_datetime = input.get("datetime", datetime_str)
        system_prompt = input.get("system_prompt", self.system_prompt)
        system_prompt = f"{system_prompt}\nCurrent datetime: {current_datetime}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        self._logger.tool_log(f"[{self.tool_name}], System Prompt: [{system_prompt}]")
        self._logger.tool_log(f"[{self.tool_name}], Prompt: [{prompt}]")
        response = await self.client.chat.completions.create(
            model=self.llm_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
            response_format={"type": self.response_format},
        )
        content = response.choices[0].message.content
        self._logger.tool_log(f"[{self.tool_name}], Response from OpenAI: [{content}]")
        response_message = Message(content={"output": content})
        return response_message


class OpenAIEmbeddingTool(Embedder):
    """The embedder is a tool to generate the embedding for the input using OpenAI embedding RESTful API.

    The url of the RESTful API is https://api.openai.com/v1/embeddings.
    Details can be seen from: https://platform.openai.com/docs/api-reference/embeddings.

    Note: This class can be deprecated once the retrieval based tooling are implemented.
    """

    tool_name: str = "openai-embedding"

    descriptions: List[str] = [
        "This tool is used to generate the embedding for the input.",
        "This tool is used to quantify the semantic meaning of the input.",
        "This tool can be used to generate the embedding for the input using OpenAI embedding.",
        "This tool can be used to generate the embedding for the input using OpenAI's text-embedding-3-small model.",
        "This tool can be used to generate the embedding for the input using OpenAI's text-embedding-3-large model.",
    ]

    url: str = "https://api.openai.com/v1/embeddings"
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="The model to generate the embedding. Choices: text-embedding-3-small, text-embedding-3-large",
    )
    embedding_restful_tool: BaseTool = RestAPITool()

    async def _embedding(self, input: List[str]) -> List[List[float]]:
        """Generate the embedding for the input using OpenAI embedding."""
        # Check OpenAI API key
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key is not set.")
        # Craft the message for the RESTful API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
        }
        params = {
            "model": self.embedding_model,
            "input": input,
            "dimension": self.embed_dim,
        }
        input_message = Message(
            content={
                "url": self.url,
                "method": "POST",
                "headers": headers,
                "params": params,
                "dimensions": self.embed_dim,
            }
        )
        # Call the RESTful API
        output_message = await self.embedding_restful_tool(input_message)
        response = output_message.content.get("response", {})
        # Check for error in the response
        if "error" in response or output_message.content.get("status_code") != 200:
            error_message = response.get("error", "Failed to retrieve embeddings")
            raise Exception(error_message)
        # Reduce the dimension of the embedding
        embedding_list = [entry.get("embedding", []) for entry in response.get("data", [])]
        embeddings: List[List[float]] = [embedding[: self.embed_dim] for embedding in embedding_list]
        return embeddings


class OpenAICodeGenerationTool(CodeGenerationTool):
    """Use OpenAI to generate code snippets based on the input prompt."""

    tool_name: str = "open-ai-code-generation"

    def _set_llm_client(self):
        model_name = os.environ.get("CODEGEN_MODEL_NAME", os.environ.get("MODEL_NAME", "gpt-3.5-turbo"))
        self._llm_tool = OpenAIChatTool(model_name=model_name)


class AnthropicClaudeTool(LLMTool):
    """AnthropicClaudeTool is a bridge to Anthropic's Claude API.
    The tool can be initialized with system_prompt, max_tokens, and temperature.
    The input message of this tool should contain a "prompt", and optionally a "system_prompt".
    The "system_prompt" in the input message will override the default system_prompt.
    The tool will return a message with key "answer" with the response from the Claude API.
    """

    model_config = {
        "arbitrary_types_allowed": True,  # Allow arbitrary types
    }
    tool_name: str = "anthropic-claude"
    llm_name: str = Field(default="claude-3-5-sonnet-20240620", description="The name of the Claude model.")
    descriptions: List[str] = [
        "This tool is used to chat with Anthropic's Claude language model.",
        "This tool can be used as the orchestrator to control the conversation and problem solving.",
        "This tool can be used to breakdown the problem into smaller parts and solve them.",
        "This tool can be used to generate the response from the chat.",
        "This tool can be used to generate the code of new tools.",
        "This tool can do simple calculation.",
        "Simple calculator that does basic arithmetic calculation.",
    ]

    client: anthropic.Client = Field(default=None)
    system_prompt: str = Field(default="You are a helpful AI assistant.")
    max_tokens: int = Field(default=4000)
    temperature: float = Field(default=0.7)
    stop: str = Field(default=None)
    response_format: str = Field(default="text", description="The format of the response from the chat.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._logger = Logger(__file__)
        self._set_client()

    def _set_client(self):
        """Set the client for the language model."""
        self.client = anthropic.Anthropic()

    def input_spec(self) -> List[Param]:
        """Return the input specification of the tool."""
        return [
            Param(
                name="system_prompt",
                type="str",
                required=False,
                example="You are a helpful AI assistant.",
                description="The system prompt for the chat.",
            ),
            Param(
                name="input",
                type="str",
                required=True,
                example="hello, how are you?",
                description="The prompt for the chat.",
            ),
            Param(
                name="max_tokens",
                type="int",
                required=False,
                example="1500",
                description="The maximum number of tokens for the chat.",
            ),
            Param(
                name="temperature",
                type="float",
                required=False,
                example="0.7",
                description="The temperature for the chat.",
            ),
        ]

    def output_spec(self) -> List[Param]:
        """Return the output specification of the tool."""
        return [
            Param(
                name="output",
                type="str",
                required=True,
                example="I'm good, how are you?",
                description="The response from the chat.",
            ),
        ]

    async def _invoke(self, input: Message) -> Message:
        """Execute the tool and return the result."""
        prompt = input.get("input", "")
        system_prompt = input.get("system_prompt", self.system_prompt)
        prompt = f"{system_prompt}\n{prompt}"
        temperature = input.get("temperature", self.temperature)
        max_tokens = input.get("max_tokens", self.max_tokens)
        messages = [
            {"role": "user", "content": prompt},
        ]
        response = self.client.messages.create(
            model=self.llm_name,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        content = response.content[0].text
        self._logger.tool_log(f"[{self.tool_name}], System Prompt: [{system_prompt}]")
        self._logger.tool_log(f"[{self.tool_name}], Prompt: [{prompt}]")
        self._logger.tool_log(f"[{self.tool_name}], Response from Claude: [{content}]")
        response_message = Message(content={"output": content})
        return response_message
