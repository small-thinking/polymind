import random
from typing import List

from pydantic import Field

from polymind.core.message import Message
from polymind.core.tool import BaseTool, Param


class PersonalFinanceTool(BaseTool):

    tool_name: str = "personal-finance-tool"
    descriptions: List[str] = Field(
        [
            "The tool to help you manage your personal finance.",
            "The personal finance tool to help you manage your money.",
            "The tool to help you track your expenses and income.",
            "The tool will help retrieve your account balance.",
        ],
        description="The descriptions of the tool.",
    )

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="input",
                type="str",
                required=True,
                description="The finance related requirement to be fulfilled by the tool.",
                example="How much money do I have in my account?",
            )
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="output",
                type="str",
                required=True,
                description="The response to the finance related requirement.",
                example="You have $10000 in your account.",
            )
        ]

    async def _execute(self, input: Message) -> Message:
        random_balance = random.randint(20000, 40000)
        return Message(content={"output": f"You have ${random_balance} in your account."})
