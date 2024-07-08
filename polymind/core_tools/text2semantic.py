"""This tool allows you to run SPARQL queries on a graph created from an ontology and data files.
"""

import re
from typing import List

import aiofiles
from rdflib import Graph
from rdflib.plugins.sparql.processor import Result

from polymind.core.logger import Logger
from polymind.core.message import Message
from polymind.core.tool import BaseTool, Param
from polymind.core_tools.llm_tool import AnthropicClaudeTool, OpenAIChatTool


class SemanticWebQueryTool(BaseTool):

    def __init__(self, tool_name: str = "semantic-web-query", *args, **kwargs):
        descriptions = [
            "Run SPARQL queries on a graph created from an ontology and data files.",
            "Query RDF graphs using SPARQL.",
            "Execute SPARQL queries on RDF graphs.",
        ]
        super().__init__(tool_name=tool_name, descriptions=descriptions, *args, **kwargs)
        self._logger = Logger(__name__)
        self._llm_tool = OpenAIChatTool()
        # self._llm_tool = AnthropicClaudeTool()
        self._sparqk_gen_prompt_template = """
            # SPARQL Query Generation
            Your task is to generate SPARQL queries based on natural language requests.
            You will be provided with information about the ontology and data structure,
            followed by examples of natural language requests and their corresponding SPARQL queries.
            Then, you will be given a new natural language request for which you should generate the appropriate SPARQL query.
            
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

    def input_spec(self) -> List[Param]:
        return [
            Param(name="nlp_query", type="str", required=True, description="NLP query to execute on the RDF graph."),
        ]

    def output_spec(self) -> List[Param]:
        return [
            Param(name="results", type="Result", required=True, description="Query results."),
        ]

    async def load_jsonld(self, file_path: str) -> str:
        """
        Asynchronously reads a JSON-LD file and returns its content as a string.

        Args:
            file_path (str): Path to the JSON-LD file.

        Returns:
            str: Content of the JSON-LD file as a string.
        """
        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
        return content

    async def load_nt(self, file_path: str) -> str:
        """
        Asynchronously reads an N-Triples file and returns its content as a string.

        Args:
            file_path (str): Path to the N-Triples file.

        Returns:
            str: Content of the N-Triples file as a string.
        """
        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
        return content

    async def create_graph(self, ontology_path: str, data_path: str) -> Graph:
        """
        Creates an RDF graph by loading ontology from a JSON-LD file and data from an N-Triples file.

        Args:
            ontology_path (str): Path to the JSON-LD ontology file.
            data_path (str): Path to the N-Triples data file.

        Returns:
            Graph: The RDF graph containing the ontology
        """
        g = Graph()

        # Load ontology
        ontology_data = await self.load_jsonld(ontology_path)
        g.parse(data=ontology_data, format="json-ld")

        # Load data
        nt_data = await self.load_nt(data_path)
        g.parse(data=nt_data, format="nt")

        return g

    async def ask_graph(self, graph: Graph, nlp_query: str) -> Result:
        """
        Executes an NLP query on the given RDF graph and returns the results.

        Args:
            graph (Graph): The RDF graph to query.
            nlp_query (str): The NLP query string.

        Returns:
            Result: Query results.
        """
        # Convert NLP query to SPARQL query using LLM
        ontology_str = graph.serialize(format="json-ld")
        sparql_gen_prompt = self._sparqk_gen_prompt_template.format(nlp_query=nlp_query, ontology_str=ontology_str)
        sparql_message = Message(content={"input": sparql_gen_prompt})
        sparql_response = await self._llm_tool(sparql_message)
        sparql_response_str = sparql_response.content["output"]
        match = re.search(r"```(?:data|sparql)?\s*\n([\s\S]*?)\n```", sparql_response_str, re.DOTALL)
        if match:
            sparql_query = match.group(1)
            self._logger.info(f"Generated SPARQL query:\n{sparql_query}")
        else:
            raise ValueError(f"SPARQL query generation failed. Response:\n\n{sparql_response_str}")

        return self.query_graph(graph=graph, query=sparql_query)

    def query_graph(self, graph: Graph, query: str) -> Result:
        """
        Executes a SPARQL query on the given RDF graph and returns the results.

        Args:
            graph (Graph): The RDF graph to query.
            query (str): The SPARQL query string.

        Returns:
            Result: Query results.
        """
        return graph.query(query)
