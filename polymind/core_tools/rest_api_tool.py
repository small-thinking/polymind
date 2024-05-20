"""RESTful API tool is a very generic tool that can be used to call any web service.

"""

import os
from typing import Dict, List

import aiohttp
from pydantic import Field

from polymind.core.logger import Logger
from polymind.core.message import Message
from polymind.core.tool import BaseTool, Param


class RestAPITool(BaseTool):
    tool_name: str = "rest-api-tool"
    descriptions: List[str] = [
        "This tool is used to call any RESTful API.",
        "This tool can be used to call any web service.",
        "This tool can be used to call any HTTP service.",
        "This tool can be used to call any web API.",
    ]

    param_spec: Dict[str, str] = Field(
        {}, description="The parameters spec to use the Restful API. List of param names and types."
    )

    def __init__(self, param_spec: Dict[str, str] = None):
        super().__init__()
        self.param_spec = param_spec or {}

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="url",
                type="str",
                required=True,
                description="The URL of the web service.",
                example="https://example.com/api/data",
            ),
            Param(
                name="method",
                type="str",
                required=True,
                description="HTTP method to use for the request.",
                example="GET",
            ),
            Param(
                name="headers",
                type="Dict[str, str]",
                required=True,
                description="HTTP headers for the request.",
                example='{"Content-Type": "application/json"}',
            ),
            Param(
                name="params",
                type="Dict[str, str]",
                required=True,
                description="URL query parameters.",
                example='{"query": "value"}',
            ),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="status_code",
                type="int",
                required=True,
                description="HTTP status code of the response.",
                example="200",
            ),
            Param(
                name="output",
                type="Dict[str, str]",
                required=False,
                description="The parsed response body. Can be a dictionary if JSON was returned or a string otherwise.",
                example='{"data": "value"}',
            ),
        ]

    def _construct_param(self, input: Message) -> Dict[str, str]:
        """Construct the parameters for the request. Add the remaining fields from the input to the params."""
        params = input.get("params", {})
        for key, value in input.content.items():
            if key not in ["url", "method", "headers", "body", "params"]:
                params[key] = input.get(key)
        # Validate that all required fields are present
        for key in self.param_spec.keys():
            if key not in params:
                raise ValueError(f"Missing required field: {key}")
        # Convert all booleans to str
        for key, value in params.items():
            if isinstance(value, bool):
                params[key] = str(value)
        return params

    def _insert_default_params(self, input: Message) -> None:
        """Insert default parameters to the input message."""
        pass

    def _post_process_response(self, response: Message) -> Message:
        """Post-process the response before returning it."""
        return response

    async def _execute(self, input: Message) -> Message:
        self._insert_default_params(input)
        url = input.get("url", "")
        method = input.get("method", "GET").upper()
        headers = input.get("headers", {})
        params = self._construct_param(input=input)

        # Determine if we need to send the request body as JSON or as form data
        if "application/json" in headers.values():
            send_json = True
        else:
            send_json = False

        async with aiohttp.ClientSession() as session:
            request_args = {
                "method": method,
                "url": url,
                "headers": headers,
                "json": params if send_json else None,
            }
            async with session.request(**request_args) as response:
                status_code = response.status
                try:
                    response_data = await response.json()
                except Exception:
                    response_data = await response.text()

                response_message = Message(content={"status_code": status_code, "response": response_data})
                return self._post_process_response(response_message)


class TavilyRestAPITool(RestAPITool):
    """Call the Tavily API to search for answers to questions."""

    tool_name: str = "tavily-rest-api-tool"
    descriptions: List[str] = [
        "Search engine to search for external information.",
        "Retrieve for information on the internet for timely information.",
        "Retrieve external search engine to search for public information.",
        "Search latest information on the internet.",
        "Search up-to-date information on the internet.",
        "Look up the world wide web for information.",
    ]

    def __init__(self):
        param_spec = {
            "api_key": "str",
            "query": "str",
            "max_results": "int",
        }
        super().__init__(param_spec=param_spec)
        self._logger = Logger(__file__)

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="query",
                type="str",
                required=True,
                description="The question to search for.",
                example="The search keywords.",
            ),
        ]

    def _insert_default_params(self, input: Message) -> None:
        input.set("include_answer", True)
        input.set("url", "https://api.tavily.com/search")
        input.set("headers", {"Content-Type": "application/json"})
        input.set("method", "POST")
        input.set("api_key", os.environ.get("TAVILY_API_KEY"))
        input.set("max_results", 10)

    def _post_process_response(self, response: Message) -> Message:
        """Only return the answer in the field "output"."""
        status_code = response.get("status_code")
        if status_code != 200:
            self._logger.error(f"Received status code {status_code} from Tavily API.")
            return response
        else:
            answer = response.get("response", {}).get("answer", "")
            response_message = Message(content={"status_code": status_code, "output": answer})
            return response_message
