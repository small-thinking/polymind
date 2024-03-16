"""
Run the test with the following command:
    poetry run pytest tests/polymind/tools/test_rest_api_tool.py
"""

import pytest
from aioresponses import aioresponses

from polymind.core.message import Message
from polymind.tools.rest_api_tool import RestAPITool


class TestRestAPITool:
    @pytest.fixture
    def rest_api_tool(self):
        return RestAPITool()

    @pytest.mark.asyncio
    async def test_query_articles(self, rest_api_tool):
        with aioresponses() as m:
            # Mocking the API response
            m.post(
                "https://wenling-production.up.railway.app/query-article/",
                payload={
                    "message": "Query successful",
                    "articles": [
                        {
                            "title": "Design2Code: How Far Are We From Automating Front-End Engineering?",
                            "url": {
                                "content": "https://arxiv.org/pdf/2403.03163.pdf",
                                "link": None,
                            },
                            "tags": ["LLM", "Deep Learning", "Design2Code"],
                        }
                    ],
                },
                status=200,
            )

            input_message = Message(
                content={
                    "url": "https://wenling-production.up.railway.app/query-article/",
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer correct-token",
                    },
                    "body": {
                        "start_date": "2024-03-12",
                        "end_date": "2024-03-17",
                        "tags": ["Design2Code"],
                    },
                }
            )

            actual_output = await rest_api_tool(input_message)
            assert actual_output.content["status_code"] == 200, "Expected HTTP status code 200"
            assert (
                actual_output.content["response"]["message"] == "Query successful"
            ), "Expected success message in response"
            assert "articles" in actual_output.content["response"], "Expected articles in response"

    @pytest.mark.asyncio
    async def test_query_articles_http_error(self, rest_api_tool):
        with aioresponses() as m:
            # Simulate an HTTP error response
            m.post("https://wenling-production.up.railway.app/query-article/", status=500)

            input_message = Message(
                content={
                    "url": "https://wenling-production.up.railway.app/query-article/",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body": {"start_date": "2024-03-12", "end_date": "2024-03-17"},
                }
            )

            actual_output = await rest_api_tool(input_message)
            assert actual_output.content["status_code"] == 500, "Expected HTTP status code 500"

    @pytest.mark.asyncio
    async def test_query_articles_bad_request(self, rest_api_tool):
        with aioresponses() as m:
            # Simulate a bad request response with a specific error message
            m.post(
                "https://wenling-production.up.railway.app/query-article/",
                payload={"error": "Bad request"},
                status=400,
            )

            input_message = Message(
                content={
                    "url": "https://wenling-production.up.railway.app/query-article/",
                    "method": "POST",
                    "headers": {"Content-Type": "application/json"},
                    "body": {},  # Missing required fields
                }
            )

            actual_output = await rest_api_tool(input_message)
            assert actual_output.content["status_code"] == 400, "Expected HTTP status code 400"
            assert actual_output.content["response"] == {"error": "Bad request"}, "Expected error message in response"

    @pytest.mark.asyncio
    async def test_missing_start_date(self, rest_api_tool):
        with aioresponses() as m:
            # Mocking the API response for missing start_date
            m.post(
                "https://wenling-production.up.railway.app/query-article/",
                payload={"error": "Missing required field: start_date"},
                status=400,
            )

            input_message = Message(
                content={
                    "url": "https://wenling-production.up.railway.app/query-article/",
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json",
                        "Authorization": "Bearer correct-token",
                    },
                    "body": {
                        # "start_date" intentionally omitted to simulate the error
                        "end_date": "2024-03-17",
                        "tags": ["Design2Code"],
                    },
                }
            )

            actual_output = await rest_api_tool(input_message)
            assert actual_output.content["status_code"] == 400, "Expected HTTP status code 400 for missing start_date"
            assert "error" in actual_output.content["response"], "Expected error message in response"
            assert (
                actual_output.content["response"]["error"] == "Missing required field: start_date"
            ), "Expected specific error message for missing start_date"
