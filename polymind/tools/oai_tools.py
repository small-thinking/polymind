"""
This file contains the necessary tools of using OpenAI models.
For now, it contains two tools: OpenAIChatTool and OpenAIEmbeddingTool.
"""

import os
from typing import List

import numpy as np
from openai import AsyncOpenAI
from pydantic import Field

from polymind.core.embedder import Embedder
from polymind.core.message import Message
from polymind.core.tool import BaseTool, Param
from polymind.tools.rest_api_tool import RestAPITool


class OpenAIChatTool(BaseTool):
    """OpenAITool is a bridge to OpenAI APIs.
    The tool can be initialized with llm_name, system_prompt, max_tokens, and temperature.
    The input message of this tool should contain a "prompt", and optionally a "system_prompt".
    The "system_prompt" in the input message will override the default system_prompt.
    The tool will return a message with the response from the OpenAI chat API.
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
    ]
    client: AsyncOpenAI = Field(default=None)
    llm_name: str = Field(default="gpt-3.5-turbo")
    system_prompt: str = Field(default="You are a helpful AI assistant.")
    max_tokens: int = Field(default=1500)
    temperature: float = Field(default=0.7)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
                example="You are a helpful AI assistant.",
                description="The system prompt for the chat.",
            ),
            Param(
                name="prompt",
                type="str",
                example="hello, how are you?",
                description="The prompt for the chat.",
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
                name="response",
                type="str",
                example="I'm good, how are you?",
                description="The response from the chat.",
            ),
        ]

    async def _execute(self, input: Message) -> Message:
        """Execute the tool and return the result.
        The derived class must implement this method to define the behavior of the tool.

        Args:
            input (Message): The input to the tool carried in a message.

        Returns:
            Message: The result of the tool carried in a message.
        """
        prompt = input.get("prompt", "")
        system_prompt = input.get("system_prompt", self.system_prompt)
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        response = await self.client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content
        response_message = Message(content={"response": content})
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

    async def _embedding(self, input: List[str]) -> np.ndarray:
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
        body = {
            "model": self.embedding_model,
            "input": input,
        }
        input_message = Message(
            content={
                "url": self.url,
                "method": "POST",
                "headers": headers,
                "body": body,
            }
        )
        # Call the RESTful API
        output_message = await self.embedding_restful_tool(input_message)
        response = output_message.content.get("response", {})
        # Check for error in the response
        if "error" in response or output_message.content.get("status_code") != 200:
            error_message = response.get("error", "Failed to retrieve embeddings")
            raise Exception(error_message)
        embeddings: List[List[float]] = [entry.get("embedding", []) for entry in response.get("data", [])]
        return np.array(embeddings)
