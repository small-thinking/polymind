"""RESTful API tool is a very generic tool that can be used to call any web service.

"""

from typing import List

import aiohttp

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

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="url",
                type="str",
                description="The URL of the web service.",
                example="https://example.com/api/data",
            ),
            Param(
                name="method",
                type="str",
                description="HTTP method to use for the request.",
                example="GET",
            ),
            Param(
                name="headers",
                type="Dict[str, str]",
                description="HTTP headers for the request.",
                example='{"Content-Type": "application/json"}',
            ),
            Param(
                name="body",
                type="Union[Dict[str, Any], str]",
                description="""The body of the request for POST/PUT methods.
                Can be a dictionary for JSON or a string for form-encoded data.
                """,
                example='{"key": "value"}',
            ),
            Param(
                name="params",
                type="Dict[str, str]",
                description="URL query parameters.",
                example='{"query": "value"}',
            ),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="status_code",
                type="int",
                description="HTTP status code of the response.",
                example="200",
            ),
            Param(
                name="response",
                type="Union[Dict[str, Any], str]",
                description="The parsed response body. Can be a dictionary if JSON was returned or a string otherwise.",
                example='{"data": "value"}',
            ),
        ]

    async def _execute(self, input: Message) -> Message:
        url = input.get("url", "")
        method = input.get("method", "GET").upper()
        headers = input.get("headers", {})
        body = input.get("body", {})
        params = input.get("params", {})

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
                "params": params,
                "json": body if send_json else None,
                "data": body if not send_json else None,
            }
            async with session.request(**request_args) as response:
                status_code = response.status
                try:
                    response_data = await response.json()
                except Exception:
                    response_data = await response.text()

                return Message(content={"status_code": status_code, "response": response_data})
