# noqa: D104
"""PolyMind - A customizable collaborative multi-agent framework."""

__version__ = "0.0.61"

# Expose the core classes
from .core.agent import Agent
from .core.message import Message
from .core.task import AtomTask, BaseTask, CompositeTask, SequentialTask
from .core.tool import BaseTool, LLMTool
from .core_tools.retrieve_tool import KnowledgeRetrieveTool, ToolRetriever

# from .core_tools.index_tool import KnowledgeIndexTool, ToolIndexer
from .core.logger import Logger

# Expose the tools
