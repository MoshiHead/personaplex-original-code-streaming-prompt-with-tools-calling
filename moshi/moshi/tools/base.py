"""
Base classes for the PersonaPlex tool calling framework.

Every tool must subclass BaseTool and implement execute().
Use safe_execute() at call sites — it catches all exceptions and logs them
so a failing tool never crashes the voice server.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Structured output from a single tool invocation."""

    tool_name: str
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_context_string(self) -> str:
        """Return a compact, human-readable string suitable for LLM context injection."""
        if not self.success:
            return ""  # failed tools contribute nothing to the prompt
        return self._format_data()

    def _format_data(self) -> str:
        if isinstance(self.data, str):
            return self.data.strip()

        if isinstance(self.data, list):
            lines: list[str] = []
            for item in self.data:
                if isinstance(item, dict):
                    title = item.get("title", "").strip()
                    snippet = item.get("snippet", item.get("content", "")).strip()
                    url = item.get("url", item.get("link", "")).strip()
                    parts = []
                    if title:
                        parts.append(title)
                    if snippet:
                        parts.append(snippet)
                    line = ": ".join(parts) if parts else ""
                    if url:
                        line += f" [{url}]"
                    if line:
                        lines.append(f"- {line}")
                else:
                    lines.append(f"- {item}")
            return "\n".join(lines)

        if isinstance(self.data, dict):
            return "\n".join(f"{k}: {v}" for k, v in self.data.items())

        return str(self.data).strip()


class BaseTool(ABC):
    """
    Abstract base for every PersonaPlex tool.

    Subclasses must define:
      name        — unique identifier used by ToolManager
      description — one-line description passed to the LLM if needed
      execute()   — the actual async implementation
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    async def execute(self, query: str, **kwargs) -> ToolResult:
        """Run the tool for the given query and return a ToolResult."""
        ...

    async def safe_execute(self, query: str, **kwargs) -> ToolResult:
        """
        Execute with timing, error capture, and structured logging.
        Always returns a ToolResult — never raises.
        """
        start = time.perf_counter()
        try:
            result = await self.execute(query, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            result.execution_time_ms = elapsed_ms
            status = "ok" if result.success else "error"
            logger.info(
                "tool=%s status=%s time_ms=%.0f query=%r",
                self.name,
                status,
                elapsed_ms,
                query[:120],
            )
            return result
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "tool=%s EXCEPTION time_ms=%.0f query=%r error=%r",
                self.name,
                elapsed_ms,
                query[:120],
                exc,
            )
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=None,
                error=str(exc),
                execution_time_ms=elapsed_ms,
            )

    def get_schema(self) -> dict:
        """Return a JSON-schema-style description of this tool (for future LLM-native tool use)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's query or topic to look up.",
                    }
                },
                "required": ["query"],
            },
        }
