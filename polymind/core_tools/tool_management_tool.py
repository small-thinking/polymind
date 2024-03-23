import json
import os
import re
from typing import Any, Dict, List

import faiss
import numpy as np
from pydantic import Field

from polymind.core.embedder import Embedder
from polymind.core.indexer import Indexer
from polymind.core.message import Message
from polymind.core.tool import BaseTool, Param
from polymind.core_tools.llm_tool import (LLMTool, OpenAIChatTool,
                                          OpenAIEmbeddingTool)


class ToolIndexer(Indexer):
    """Indexer for the learned tools."""

    tool_name: str = "tool-indexer"
    learned_tool_folder: str = Field(default="./knowledge/tools", description="The folder to store the learned tools.")

    embedder: Embedder = OpenAIEmbeddingTool()

    descriptions: List[str] = [
        "ToolIndexer is a tool to help find external knowledge",
        "ToolIndexer is a tool to internalize the tools.",
        "ToolIndexer is a tool to index the learned tools into the knowledge base.",
    ]

    def _extra_input_spec(self) -> List[Param]:
        return [
            Param(
                name="tool_name",
                type="str",
                description="The name of the tool to be indexed.",
                example="rest-api-tool",
            ),
            Param(
                name="descriptions",
                type="List[str]",
                description="The descriptions of the tool to be indexed.",
                example="""[
                    "This tool is used to call any RESTful API.",
                    "This tool can be used to call any web service.",
                    "This tool can be used to call any HTTP service.",
                    "This tool can be used to call any web API.",
                ]""",
            ),
            Param(
                name="tool_file_name",
                type="str",
                description="The file name of the tool. The tool will be stored under the knowledge folder.",
                example="rest_api_tool.py",
            ),
        ]

    def _create_or_update_index(self, embedding: np.ndarray):
        index_path = os.path.join(self.learned_tool_folder, "tool.index")
        if not os.path.exists(index_path):
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
        if os.path.exists(index_path):
            # Load the index if it already exists.
            index = faiss.read_index(index_path)
            # Incrementally update the index.
            index.add(embedding)
            faiss.write_index(index, index_path)
        else:
            # Create a new index if it does not exist.
            index = faiss.IndexFlatL2(embedding.shape[1])
            index.add(embedding)
            faiss.write_index(index, index_path)

    def _create_or_update_metadata(self, tool_name: str, descriptions: List[str], filename: str):
        metadata_json_path = os.path.join(self.learned_tool_folder, "tool_profiles.json")
        tool_metadata = {
            "tool_name": tool_name,
            "descriptions": descriptions,
            "file_name": filename,
        }
        existing_tools = []
        if not os.path.exists(metadata_json_path):
            os.makedirs(os.path.dirname(metadata_json_path), exist_ok=True)
        else:
            # Load the metadata json file.
            with open(metadata_json_path, "r") as f:
                existing_tools = json.load(f)
        # Append the new tool metadata to the metadata json file.
        # Note here we append a tool metadata multiple times if it is indexed multiple times.
        for _ in range(len(descriptions)):
            existing_tools.append(tool_metadata)
        # Write the metadata json file.
        with open(metadata_json_path, "w") as f:
            json.dump(existing_tools, f, indent=4)

    async def _execute(self, input_message: Message) -> Message:
        """Index the tool. Index consists of two parts:
        Embedding: The embedding is managed by FAISS.
        Metadata: The metadata is stored in a json file so it is human readable.
            The metadata contains the list of tools with their tool name, descriptions, and file name.
        """
        tool_name = input_message.content["tool_name"]
        descriptions = input_message.content["descriptions"]
        filename = input_message.content["tool_file_name"]
        embedding = await self.embedder._embedding(descriptions)
        # Save the index, if index exists, increment the index.
        self._create_or_update_index(embedding)
        # Save the actual tool file, if folder not exists, create the folder.
        self._create_or_update_metadata(tool_name=tool_name, descriptions=descriptions, filename=filename)
        return Message(content={"status": "success"})


class ToolRetriever(BaseTool):
    tool_name: str = "tool-retriever"

    descriptions: List[str] = [
        "ToolRetriever is a tool to help find external knowledge",
        "ToolRetriever is a tool to retrieve the tools.",
        "ToolRetriever is a tool to retrieve the learned tools from the knowledge base.",
    ]

    learned_tool_folder: str = Field(
        default="./knowledge/tools",
        description="The folder containing the tool index and metadata.",
    )
    top_k: int = Field(default=3, description="Number of top relevant tools to retrieve.")
    embedder: Embedder = Field(OpenAIEmbeddingTool(), description="The embedder to generate the embedding.")

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="requirement",
                type="str",
                description="Text-based requirement description for tool retrieval.",
                example="I need a tool to call REST API.",
            )
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="candidates",
                type="List[str]",
                description="List of retrieved candidate tool metadata in json string.",
                example="""[
                '{
                    "tool_name": "rest-api-tool",
                    "descriptions": [
                        "This tool is used to call any RESTful API.",
                        "This tool can be used to call any web service.",
                        "This tool can be used to call any web API."
                    ],
                    "file_name": "rest_api_tool.py"
                }'
                ]""",
            )
        ]

    def _find_top_k_candidates(self, query_embedding: np.ndarray) -> List[Dict[str, Any]]:
        index_path = os.path.join(self.learned_tool_folder, "tool.index")
        metadata_path = os.path.join(self.learned_tool_folder, "tool_profiles.json")
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Index or metadata file not found.")
        # Load FAISS index
        index = faiss.read_index(index_path)
        # Ensure the index is not empty
        if index.ntotal == 0:
            raise ValueError("FAISS index is empty.")
        # Query the index
        distances, indices = index.search(query_embedding.astype(np.float32), min(self.top_k, index.ntotal))
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        # Prepare candidates with their similarity scores
        candidates = []
        tool_names = set()
        for i, idx in enumerate(indices[0]):
            if idx < len(metadata) and metadata[idx]["tool_name"] not in tool_names:
                tool_metadata = metadata[idx]
                similarity_score = 1.0 / (1 + distances[0][i])  # Convert distance to similarity
                candidates.append(
                    {
                        "tool_name": tool_metadata["tool_name"],
                        "descriptions": tool_metadata["descriptions"],
                        "similarity": similarity_score,
                        "file_name": tool_metadata["file_name"],
                    }
                )
                tool_names.add(tool_metadata["tool_name"])
        return candidates

    async def _execute(self, input_message: Message) -> Message:
        requirement = input_message.content["requirement"]
        embedding = await self.embedder._embedding([requirement])
        candidates = self._find_top_k_candidates(embedding)
        return Message(content={"candidates": candidates})


class ToolCreator(BaseTool):
    tool_name: str = "tool-creator"

    llm_tool: LLMTool = Field(default=None, description="The LLM tool to generate the tool code.")

    descriptions: List[str] = [
        "ToolCreator is a tool to generate the tool code based on the requirement",
        "ToolCreator is a tool to create the tool.",
        "ToolCreator is a codegen tool to generate the tool code based on the requirement.",
    ]

    learned_tool_folder: str = Field(default="./knowledge/tools", description="The folder to store the learned tools.")

    system_prompt: str = Field(
        default="""
        You are a code generator. You are given a requirement to create a tool.
        The generated tool should inherit the below BaseTool class.
        Please put your answer into a ```python``` code block.

        --- python below ---
        # File: {{tool_name}}.py

        class BaseTool(BaseModel, ABC):
            '''The base class of the tool.
            In an agent system, a tool is an object that can be used to perform a task.
            For example, search for information from the internet, query a database,
            or perform a calculation.
            '''

            tool_name: str = Field(..., description="The name of the tool.")
            descriptions: List[str] = Field(
                ...,
                min_length=3,
                description='''The descriptions of the tool. The descriptions will be
                converted to embeddings and used to index the tool. One good practice is to
                describe the tools with the following aspects: what the tool does, and describe
                the tools from different perspectives.
                ''',
            )

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                load_dotenv(override=True)

            def __str__(self):
                return self.tool_name

            @field_validator("tool_name")
            def check_tool_name(cls, v: str) -> str:
                if not v:
                    raise ValueError("The tool_name must not be empty.")
                return v

            @field_validator("descriptions")
            def check_descriptions(cls, v: List[str]) -> List[str]:
                if len(v) < 3:
                    raise ValueError("The descriptions must have at least 3 items. The more the better.")
                return v

            def get_descriptions(self) -> List[str]:
                return self.descriptions

            async def __call__(self, input: Message) -> Message:
                '''Makes the instance callable, delegating to the execute method.
                This allows the instance to be used as a callable object, simplifying the execution of the tool.

                Args:
                    input (Message): The input message to the tool.

                Returns:
                    Message: The output message from the tool.
                '''
                return await self._execute(input)

            def get_spec(self) -> str:
                '''Return the input and output specification of the tool.

                Returns:
                    Tuple[List[Param], List[Param]]: The input and output specification of the tool.
                '''
                input_json_obj = []
                for param in self.input_spec():
                    input_json_obj.append(param.to_json_obj())
                output_json_obj = []
                for param in self.output_spec():
                    output_json_obj.append(param.to_json_obj())
                spec_json_obj = {
                    "input_message": input_json_obj,
                    "output_message": output_json_obj,
                }
                return json.dumps(spec_json_obj, indent=4)

            @abstractmethod
            def input_spec(self) -> List[Param]:
                '''Return the specification of the input parameters.'''
                pass

            @abstractmethod
            def output_spec(self) -> List[Param]:
                '''Return the specification of the output parameters.'''
                pass

            @abstractmethod
            async def _execute(self, input: Message) -> Message:
                '''Execute the tool and return the result.
                The derived class must implement this method to define the behavior of the tool.

                Args:
                    input (Message): The input to the tool carried in a message.

                Returns:
                    Message: The result of the tool carried in a message.
                '''
                pass
        --- python above ---

        In addition to any needed package, the implementation should also import the below packages:

        from polymind.core.message import Message
        from polymind.core.tool import BaseTool, Param
        """,
        description="The system prompt to generate the tool code.",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.llm_tool:
            self.llm_tool = OpenAIChatTool(llm_name="gpt-3.5-turbo", system_prompt=self.system_prompt)

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="requirement",
                type="str",
                description="Text-based requirement description for tool creation.",
                example="I need a tool to call REST API.",
            )
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="status",
                type="str",
                description="Whether the tool is successfully created and indexed.",
                example="success",
            )
        ]

    async def _execute(self, input_message: Message) -> Message:
        """Leverage the LLM tool to generate the tool code based on the requirement.

        Args:
            input_message (Message): The input message containing the requirement.
                The message should contain the "requirement" key.
        """
        requirement = input_message.content.get("requirement", "")
        if not requirement:
            raise ValueError("Requirement not provided.")
        input_message = Message(content={"prompt": requirement, "system_prompt": self.system_prompt})
        tool_code = await self.llm_tool(input=input_message)

        # Check the ```python``` code block in the tool_code and extract the source code.
        tool_code = tool_code.content["response"]
        tool_code = re.search(r"```python(.*?)```", tool_code, re.DOTALL)
        if not tool_code:
            raise ValueError("Tool code not found.")
        tool_code = tool_code.group(1).strip()
        # Save the tool code to a file.
        tool_name = re.search(r"class\s+(.*?)\(", tool_code).group(1)
        if not tool_name:
            raise ValueError("Tool name not found.")
        tool_name = tool_name.strip().lower()
        tool_file_name = f"{tool_name}.py"
        tool_file_path = os.path.join(self.learned_tool_folder, "test_learned", tool_file_name)

        if not os.path.exists(os.path.dirname(tool_file_path)):
            os.makedirs(os.path.dirname(tool_file_path), exist_ok=True)
        with open(tool_file_path, "w") as f:
            f.write(tool_code)

        # Output message
        return Message(content={"status": "success"})
