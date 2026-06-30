"""
ToolManager — central registry and orchestrator for PersonaPlex tools.

Usage pattern (server startup):
    tool_manager = ToolManager()
    tool_manager.register(SearchTool())
    tool_manager.register(CryptoTool())
    tool_manager.register(FinanceTool())

Per-request usage (in /api/augment-prompt handler):
    augmented, results = await tool_manager.augment_prompt_async(base_prompt, user_query)

Adding a new tool requires zero changes here — just implement BaseTool and
call register().
"""

import asyncio
import logging
from typing import Optional

from .base import BaseTool, ToolResult
from .intent import detect_live_intent, describe_intent

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Guidance injected into every augmented system prompt.
# Follows the "Information: <facts>" in-distribution pattern the PersonaPlex
# model was trained on (see README "Customer Service Roles" examples).
# ──────────────────────────────────────────────────────────────────────────────

_TOOL_GUIDANCE = (
    "You have access to real-time information retrieved from live APIs. "
    "Use the data in the Information section precisely and do not invent "
    "numbers, prices, names, or dates. If the user asks about something "
    "not covered in the Information, say you don't have that specific "
    "current data and suggest they check a live source."
)


class ToolManager:
    """
    Registry and dispatcher for PersonaPlex tools.

    Thread-safe for read access after registration is complete (registration
    should happen at server startup before any requests arrive).
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        if not tool.name:
            raise ValueError(f"Tool {type(tool).__name__} has no name defined.")
        self._tools[tool.name] = tool
        logger.info("tool registered: %s — %s", tool.name, tool.description)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    # ── Intent → tool selection ───────────────────────────────────────────────

    def detect_tools(self, query: str) -> list[str]:
        """
        Return ordered list of tool names to invoke for this query.
        Empty list means no live data is needed.
        """
        logger.debug("intent check: %s", describe_intent(query))
        needs_live, categories = detect_live_intent(query)
        if not needs_live:
            return []

        tool_names: list[str] = []
        for cat in categories:
            if cat in self._tools and cat not in tool_names:
                tool_names.append(cat)

        # Fallback: if intent says "search" but no search tool registered, skip
        return [t for t in tool_names if t in self._tools]

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute_tools(
        self, tool_names: list[str], query: str
    ) -> list[ToolResult]:
        """Run selected tools concurrently and collect results."""
        tasks = [
            self._tools[name].safe_execute(query)
            for name in tool_names
            if name in self._tools
        ]
        if not tasks:
            return []
        results = await asyncio.gather(*tasks)
        return list(results)

    # ── Context building ──────────────────────────────────────────────────────

    def build_context_block(self, results: list[ToolResult]) -> str:
        """
        Combine tool results into a single context string for the LLM.
        Failed tools are silently omitted; successful ones are separated by newlines.
        """
        parts: list[str] = []
        for result in results:
            ctx = result.to_context_string()
            if ctx:
                parts.append(ctx)
        return "\n".join(parts)

    def augment_prompt(
        self, base_prompt: str, query: str
    ) -> tuple[str, list[ToolResult]]:
        """
        Synchronous convenience wrapper.  Use augment_prompt_async in async contexts.
        Creates a temporary event loop — only safe to call from a non-async context.
        """
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.augment_prompt_async(base_prompt, query)
            )
        finally:
            loop.close()

    async def augment_prompt_async(
        self, base_prompt: str, query: str
    ) -> tuple[str, list[ToolResult]]:
        """
        Detect tools, execute them concurrently, and return an augmented
        system prompt together with the raw ToolResult list.

        Returns
        -------
        (augmented_prompt, tool_results)
            augmented_prompt — ready to pass as text_prompt to the WebSocket
            tool_results     — raw results for logging / API response
        """
        tool_names = self.detect_tools(query)
        if not tool_names:
            logger.debug("no tools needed for query %r", query[:80])
            return base_prompt, []

        logger.info(
            "invoking tools %s for query %r", tool_names, query[:120]
        )
        results = await self.execute_tools(tool_names, query)
        context = self.build_context_block(results)

        if not context:
            # All tools failed — return base prompt unchanged
            logger.warning(
                "all tools returned empty context for query %r; using base prompt",
                query[:80],
            )
            return base_prompt, results

        augmented = _build_augmented_prompt(base_prompt, context)
        return augmented, results


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_augmented_prompt(base_prompt: str, tool_context: str) -> str:
    """
    Combine the user's base system prompt with tool context.

    Output format follows the in-distribution PersonaPlex prompt pattern:
      "<role sentence>. <guidance>. Information: <facts>"

    This is consistent with the Customer Service examples in README.md and
    with how the model ingests grounding information via `step_system_prompts`.
    """
    base = base_prompt.strip() if base_prompt else ""

    if not base:
        base = "You are a helpful, knowledgeable real-time voice assistant."

    # Avoid duplicating guidance if it was already injected
    if _TOOL_GUIDANCE[:40] in base:
        return f"{base} Information: {tool_context}"

    return f"{base} {_TOOL_GUIDANCE} Information: {tool_context}"
