import json
import re
from typing import Any, Dict, List

from polymind.core.tool import BaseTool, LLMTool, Message, Param
from polymind.core.utils import Logger


class SemanticWebIndexer(BaseTool):
    """The SemanticWebIndexer tool processes semantic web data based on a description and input data.
    It is able to generate the ontology and semantic web data in various formats,
    based on the input data and description.
    """

    descriptions: List[str] = [
        "Process semantic web data based on a description and input data.",
        "Generate ontologies and semantic web data in various formats.",
        "Convert input data to semantic web format using custom ontologies.",
    ]

    def __init__(self, llm_tool: LLMTool, tool_name: str = "semantic-web-indexer", *args, **kwargs):
        descriptions = [
            "Process semantic web data based on a description and input data.",
            "Generate ontologies and semantic web data in various formats.",
            "Convert input data to semantic web format using custom ontologies.",
        ]
        super().__init__(tool_name=tool_name, descriptions=descriptions, *args, **kwargs)
        self._logger = Logger(__name__)
        self._llm_tool = llm_tool
        self._ontology_gen_propmt_template = """
            Your task is to generate an ontology based on the given description.
            The output format of the ontology should be {ontology_data_format}.

            The description is as below. If not otherwise specified, the namespace will be {namespace}:
            {ontology_description}

            Please generate the ontology and provide the output in a blob wrapped in ```data```.
        """
        self._data_gen_prompt_template = """
            Your task is to convert the given data to semantic web format based on the given ontology.
            The output format of the data should be {data_output_format}.

            The ontology is as below. The raw data may contain irrelevant information. We can ignore them.
            {ontology_str}

            The input data is:
            {input_data}

            Please generate the semantic web data and provide the output in a blob wrapped in ```data```.
        """

    def input_spec(self) -> List[Param]:
        return [
            Param(
                name="ontology_description",
                type="str",
                required=True,
                description="Description of the ontology and data structure",
            ),
            Param(
                name="data_input_path",
                type="str",
                required=True,
                description="Path to the JSON or JSONL formatted data file to be processed",
            ),
            Param(
                name="ontology_data_format",
                type="str",
                required=False,
                description="Data format for the ontology (default: 'jsonld')",
            ),
            Param(
                name="ontology_output_path",
                type="str",
                required=False,
                description="Output file path for the ontology. Optional.",
            ),
            Param(
                name="data_output_format",
                type="str",
                required=False,
                description="Output format for the semantic data (default: 'nt')",
            ),
            Param(
                name="data_output_path",
                type="str",
                required=False,
                description="Output file path for the data. If not specified, the data will be returned as a string.",
            ),
            Param(
                name="namespace",
                type="str",
                required=False,
                description="Namespace for the ontology (default: 'http://sw.com')",
            ),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(
                name="ontology_output",
                type="str",
                required=True,
                description="The generated ontology in RDF format",
            ),
            Param(
                name="knowledge_output",
                type="str",
                required=True,
                description="If output_file_path specified, return the path. Otherwise the semantic data as a string.",
            ),
        ]

    async def _execute(self, input_message: Message) -> Message:
        # Process input parameters and set default values
        ontology_description = input_message.content["ontology_description"]
        data_input_path = input_message.content["data_input_path"]
        ontology_data_format = input_message.content.get("ontology_data_format", "jsonld")
        ontology_output_path = input_message.content.get("ontology_output_path")
        data_output_format = input_message.content.get("data_output_format", "nt")
        data_output_path = input_message.content.get("data_output_path")
        namespace = input_message.content.get("namespace", "http://sw.com")

        # Generate ontology
        ontology_str = await self.generate_ontology(
            description=ontology_description, namespace=namespace, format=ontology_data_format
        )
        # Write to file if output path is specified.
        if ontology_output_path:
            with open(ontology_output_path, "w") as f:
                f.write(ontology_str)
            self._logger.info(f"Ontology written to {ontology_output_path}")
            ontology_output = f"Ontology written to {ontology_output_path}"
        else:
            ontology_output = ontology_str

        # Process data and generate semantic web data
        semantic_data = await self.generate_semantic_data(
            ontology_description, ontology_str, data_input_path, data_output_format, namespace
        )

        # Write semantic data to file if output path is specified, otherwise return as string
        if data_output_path:
            with open(data_output_path, "w") as f:
                f.write(semantic_data)
            knowledge_output = f"Semantic data written to {data_output_path}"
            self._logger.info(f"Semantic data written to {data_output_path}")
        else:
            knowledge_output = semantic_data

        return Message(content={"ontology_output": ontology_output, "knowledge_output": knowledge_output})

    async def generate_ontology(self, description: str, namespace: str, format: str) -> str:
        """Leverage the LLM to generate the ontology.

        Args:
            description (str): The description of the ontology.
            namespace (str): The namespace for the ontology.
            format (str): The output format for the ontology.

        Returns:
            str: The generated ontology in string.
        """
        ontology_generation_prompt = self._ontology_gen_propmt_template.format(
            ontology_data_format=format, namespace=namespace, ontology_description=description
        )
        ontology_message = Message(content={"input": ontology_generation_prompt})
        ontology_response = await self._llm_tool(ontology_message)
        ontology_response_str = ontology_response.content["output"]
        match = re.search(r"```(?:data)?\s*\n([\s\S]*?)\n```", ontology_response_str, re.DOTALL)
        if match:
            ontology_str = match.group(1)
        else:
            raise ValueError(f"Ontology generation failed. Response:\n\n{ontology_response_str}")

        return ontology_str

    async def generate_semantic_data(
        self,
        description: str,
        ontology_str: str,
        data_input_path: str,
        output_format: str = "nt",
        namespace: str = "http://sw.com",
    ) -> str:
        """Convert the raw data to semantic web data based on the ontology.

        Args:
            description (str): The description of the ontology.
            ontology_str (str): The generated ontology in string.
            data_input_path (str): The path to the JSON or JSONL formatted data file to be processed.
            output_format (str): The output format for the semantic data.
            namespace (str): The namespace for the ontology.

        Returns:
            str: The generated semantic web data in string.
        """
        # Load the data from file.
        data = self.load_jsonl(data_input_path)

        # Batch process 10 records at a time.
        batch_size = 10
        semantic_data = []

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            batch_str = json.dumps(batch)

            # Use LLM to convert to RDF as specified format.
            data_gen_prompt = self._data_gen_prompt_template.format(
                data_output_format=output_format, ontology_str=ontology_str, input_data=batch_str
            )
            data_message = Message(content={"input": data_gen_prompt})
            data_response = await self._llm_tool(data_message)
            data_response_str = data_response.content["output"]

            # Extract the generated semantic data
            match = re.search(r"```(?:data)?\s*\n([\s\S]*?)\n```", data_response_str, re.DOTALL)
            if match:
                semantic_data.append(match.group(1))
            else:
                self._logger.warning(
                    f"Semantic data generation failed for batch {i//batch_size}. Response:\n\n{data_response_str}"
                )

        # Combine all batches
        combined_semantic_data = "\n".join(semantic_data)

        return combined_semantic_data

    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r") as file:
            return [json.loads(line) for line in file]
