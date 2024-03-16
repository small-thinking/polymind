## Knowledge

Knowledge is all we need.
The traditional LLM-centric approach only treat the facts and data as knowledge. However,
everything can be represented as the knowledge, including how to use the tools, how to create the tools,
and the reasoning methodology themselves.

This is similar to the intelligence of human being. The differences between animal and human being is not
how to use the existing tool, how to leverage external data to enhance our knowldge (how we currently use RAG),
but operating at a higher level, how to create the tools, how to create the knowledge, and how to create the reasoning methodology.

The advantage of this approach have the following benefits:
1. We don't need to create tool one by one. Let LLM to do that labor work for us.
2. The library is lightweight, because there is no need to keep an endless list of tools/pipelines due to reason 1.

With this design, we only need 4 tools to create all the tools and knowledge we need:
1. LLM: LLM is the tool we used to do basic reasoning.
2. Code interpreter: Code interpreter is used to validate the tools we created.
3. Search engine: This helps the system to seek the knowledge externally.
4. Vector database: Vector database is used to store the knowledge we created. This can help the system to internalize the knowledge. It can also help the system to materialize the tools that can be reused in the future.