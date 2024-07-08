"""This tool allows you to run SPARQL queries on a graph created from an ontology and data files.
"""

import re
from typing import List, Tuple

from pydantic import Field
from rdflib import Graph
from rdflib.plugins.sparql.processor import Result

from polymind.core.logger import Logger
from polymind.core.message import Message
from polymind.core.tool import BaseTool, LLMTool, Param


class SemanticWebQueryTool(BaseTool):

    ontology_path: str = Field(..., description="Path to the JSON-LD ontology file.")
    data_path: str = Field(..., description="Path to the N-Triples data file.")

    def __init__(self, llm_tool: LLMTool, tool_name: str = "semantic-web-query", *args, **kwargs):
        descriptions = [
            "Run SPARQL queries on a graph created from an ontology and data files.",
            "Query RDF graphs using SPARQL.",
            "Execute SPARQL queries on RDF graphs.",
        ]
        super().__init__(tool_name=tool_name, descriptions=descriptions, *args, **kwargs)
        self._logger = Logger(__name__)
        self._llm_tool = llm_tool
        self._sparql_gen_prompt_template = """
            # SPARQL Query Generation
            Your task is to generate SPARQL queries based on natural language requests.
            You will be provided with information about the ontology and data structure,
            followed by examples of natural language requests and their corresponding SPARQL queries.
            Then, you will be given a new natural language request for which you should
            generate the appropriate SPARQL query.

            # Ontology and Data Structure
            The ontology includes the following main classes and properties:
            {ontology_str}

            # Examples
            ## Example 1:
            ### Natural Language Request: "List all items and their names."
            ### SPARQL query:
            ```sparql
            PREFIX sw: <http://sw.org/>

            SELECT ?item ?name
            WHERE {{
                ?item sw:name ?name .
            }}
            ```

            ## Example 2:
            ### Natural Language Request: "Find all items in the 'Sports' genre, including sub-genres."
            ### SPARQL query:
            ```sparql
            PREFIX sw: <http://sw.org/>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

            SELECT ?item ?itemName ?genre ?genreName
            WHERE {{
                ?item sw:name ?itemName ;
                        sw:hasGenre ?itemGenre .

                ?itemGenre rdf:type sw:Genre .
                ?itemGenre sw:name ?genreName .

                ?itemGenre (sw:parentGenre)* ?genre .

                FILTER(?genre = <http://sw.org/genre/Sports>)
            }}
            ```

            ## Example 3:
            ### Natural Language Request: "Show me the names of all genres and their parent genres."
            ### SPARQL query:
            ```sparql
            PREFIX sw: <http://sw.org/>

            SELECT ?genreName ?parentGenreName
            WHERE {{
                ?genre sw:name ?genreName .
                OPTIONAL {{
                    ?genre sw:parentGenre ?parentGenre .
                    ?parentGenre sw:name ?parentGenreName .
                }}
            }}
            ```

            # Your Task
            Now, please generate a SPARQL query for the following natural language request:
            {nlp_query}

            Please put the SPARQL query in a blob wrapped in ```data```.
        """
        # Load the semantic web data as separate graphs
        self._ontology_graph, self._data_graph = self.create_graphs(
            ontology_path=self.ontology_path, data_path=self.data_path
        )
        # Create a combined graph for querying
        self._combined_graph = self._ontology_graph + self._data_graph

    def input_spec(self) -> List[Param]:
        return [
            Param(name="nlp_query", type="str", required=True, description="NLP query to execute on the RDF graph."),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(name="results", type="List[str]", required=True, description="Query results."),
        ]

    def load_jsonld(self, file_path: str) -> str:
        """
        Asynchronously reads a JSON-LD file and returns its content as a string.

        Args:
            file_path (str): Path to the JSON-LD file.

        Returns:
            str: Content of the JSON-LD file as a string.
        """
        with open(file_path, "r") as f:
            content = f.read()
        return content

    def load_nt(self, file_path: str) -> str:
        """
        Asynchronously reads an N-Triples file and returns its content as a string.

        Args:
            file_path (str): Path to the N-Triples file.

        Returns:
            str: Content of the N-Triples file as a string.
        """
        with open(file_path, "r") as f:
            content = f.read()
        return content

    def create_graphs(self, ontology_path: str, data_path: str) -> Tuple[Graph, Graph]:
        """
        Creates separate RDF graphs for ontology and data.

        Args:
            ontology_path (str): Path to the JSON-LD ontology file.
            data_path (str): Path to the N-Triples data file.

        Returns:
            Tuple[Graph, Graph]: The ontology graph and the data graph.
        """
        ontology_graph = Graph()
        data_graph = Graph()

        # Load ontology
        ontology_data = self.load_jsonld(ontology_path)
        ontology_graph.parse(data=ontology_data, format="json-ld")

        # Load data
        nt_data = self.load_nt(data_path)
        data_graph.parse(data=nt_data, format="nt")

        return ontology_graph, data_graph

    async def ask_graph(self, nlp_query: str) -> Result:
        """
        Executes an NLP query on the combined RDF graph and returns the results.

        Args:
            nlp_query (str): The NLP query string.

        Returns:
            Result: Query results.
        """
        # Convert NLP query to SPARQL query using LLM
        ontology_str = self._ontology_graph.serialize(format="json-ld")
        sparql_gen_prompt = self._sparql_gen_prompt_template.format(nlp_query=nlp_query, ontology_str=ontology_str)
        sparql_message = Message(content={"input": sparql_gen_prompt})
        sparql_response = await self._llm_tool(sparql_message)
        sparql_response_str = sparql_response.content["output"]
        match = re.search(r"```(?:data|sparql)?\s*\n([\s\S]*?)\n```", sparql_response_str, re.DOTALL)
        if match:
            sparql_query = match.group(1)
            self._logger.info(f"Generated SPARQL query:\n{sparql_query}")
        else:
            raise ValueError(f"SPARQL query generation failed. Response:\n\n{sparql_response_str}")

        return self.query_graph(query=sparql_query)

    def query_graph(self, query: str) -> str:
        """
        Executes a SPARQL query on the combined RDF graph and returns the results.

        Args:
            query (str): The SPARQL query string.

        Returns:
            str: Query results.
        """
        results = self._combined_graph.query(query)
        result_list = []

        for row in results:
            row_str = ", ".join([str(value) for value in row])
            result_list.append(row_str)

        return result_list

    async def _execute(self, input: Message) -> Message:
        nlp_query = input.content["nlp_query"]
        results = await self.ask_graph(nlp_query=nlp_query)
        return Message(content={"results": results})
