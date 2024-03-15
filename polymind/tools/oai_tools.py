"""
This file contains the necessary tools of using OpenAI models.
"""

from typing import List
from pydantic import Field
from polymind.core.tool import BaseTool, Param
from polymind.core.message import Message
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv


class OpenAIChatTool(BaseTool):
    """OpenAITool is a bridge to OpenAI APIs.
    The tool can be initialized with llm_name, system_prompt, max_tokens, and temperature.
    The input message of this tool should contain a "prompt", and optionally a "system_prompt".
    The "system_prompt" in the input message will override the default system_prompt.
    The tool will return a message with the response from the OpenAI chat API.
    """

    class Config:
        arbitrary_types_allowed: bool = True  # Allow arbitrary types

    tool_name: str = "open-ai-chat"
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
            Param(name="prompt", type="str", description="The prompt for the chat."),
            Param(
                name="system_prompt",
                type="str",
                description="The system prompt for the chat.",
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
                name="response", type="str", description="The response from the chat."
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
