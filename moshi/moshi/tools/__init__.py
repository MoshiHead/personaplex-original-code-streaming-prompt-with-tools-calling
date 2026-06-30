"""
PersonaPlex Tool Calling Framework

Provides live, real-time information retrieval for the voice assistant.
Tools are invoked before the WebSocket conversation begins so results are
injected into the system prompt — the only insertion point compatible with
the full-duplex speech-to-speech architecture.
"""

from .base import BaseTool, ToolResult
from .manager import ToolManager
from .search import SearchTool
from .crypto import CryptoTool
from .finance import FinanceTool

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolManager",
    "SearchTool",
    "CryptoTool",
    "FinanceTool",
]
