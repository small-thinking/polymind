from typing import List
from polymind.core_tools.llm_tool import LLMTool
from polymind.core.message import Message
from polymind.core.tool import Param, Field
import aiohttp
from polymind.core.logger import Logger
import datetime
import os


class GroqLLMTool(LLMTool):
    """GroqLLMTool is a tool that calls the Groq LLM API for language model tasks."""

    model_config = {"arbitrary_types_allowed": True}
    tool_name: str = "groq-llm-api-tool"
    descriptions: List[str] = [
        "Call the Groq LLM API for language model tasks.",
        "Send a request to the Groq API to get language model completions.",
        "Use Groq API to interact with their language models.",
    ]
    llm_name: str = "llama3-70b-8192"
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
        """Initialize the async http client."""
        self._url = "https://api.groq.com/openai/v1/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', '')}",
            "Content-Type": "application/json",
        }

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
        # Send async http request to the Groq API
        params = {
            "messages": messages,
            "model": self.llm_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }
        async with aiohttp.ClientSession().post(self._url, headers=self._headers, json=params) as response:
            status_code = response.status
            response_data = await response.json()
            if status_code != 200:
                self._logger.tool_log(f"[{self.tool_name}], Error: [{response_data}]")
                response_message = Message(content={"status_code": status_code, "response": response_data})
            else:
                choices = response_data.get("choices", [{}])
                llm_message = choices[0].get("message", {})
                completion = llm_message.get("content", "")
                self._logger.tool_log(f"[{self.tool_name}], Response: [{completion}]")
                response_message = Message(content={"status_code": status_code, "output": completion})

        return response_message
