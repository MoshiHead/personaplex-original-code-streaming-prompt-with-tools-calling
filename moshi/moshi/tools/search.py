"""
Web search tool for PersonaPlex — multi-provider with automatic fallback.

Provider priority (first available API key wins):
  1. Tavily       — TAVILY_API_KEY     — best structured results for LLMs
  2. Serper       — SERPER_API_KEY     — Google search results
  3. Brave Search — BRAVE_API_KEY      — independent index, free tier available
  4. DuckDuckGo   — no key required    — always available as last resort

Install the optional duckduckgo-search package to enable the HTML DDG
fallback when the Instant Answer API returns no results:
    pip install duckduckgo-search
"""

import logging
import os

import aiohttp

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_MAX_RESULTS = 5
_TIMEOUT_S = 10
_UA = (
    "Mozilla/5.0 (PersonaPlex voice assistant; "
    "+https://github.com/NVIDIA/personaplex)"
)
_BASE_HEADERS = {"User-Agent": _UA, "Accept": "application/json"}


class SearchTool(BaseTool):
    """
    Live web search.  Automatically selects the best available provider
    based on configured API keys.

    Environment variables (all optional — DDG used if none set):
        TAVILY_API_KEY
        SERPER_API_KEY
        BRAVE_API_KEY
    """

    name = "search"
    description = (
        "Web search for live information: current events, news, sports results, "
        "people, politics, AI news, and any factual query requiring fresh data."
    )

    def __init__(self) -> None:
        self._tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
        self._serper_key = os.getenv("SERPER_API_KEY", "").strip()
        self._brave_key = os.getenv("BRAVE_API_KEY", "").strip()
        provider = (
            "Tavily" if self._tavily_key
            else "Serper" if self._serper_key
            else "Brave" if self._brave_key
            else "DuckDuckGo (no key)"
        )
        logger.info("SearchTool initialised — provider: %s", provider)

    async def execute(self, query: str, **kwargs) -> ToolResult:
        if self._tavily_key:
            return await self._tavily(query)
        if self._serper_key:
            return await self._serper(query)
        if self._brave_key:
            return await self._brave(query)
        return await self._duckduckgo(query)

    # ── Tavily ────────────────────────────────────────────────────────────────

    async def _tavily(self, query: str) -> ToolResult:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self._tavily_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": _MAX_RESULTS,
        }
        timeout = aiohttp.ClientTimeout(total=_TIMEOUT_S)
        async with aiohttp.ClientSession(headers=_BASE_HEADERS) as session:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                resp.raise_for_status()
                data = await resp.json()

        results: list[dict] = []
        if data.get("answer"):
            results.append(
                {"title": "Summary", "snippet": data["answer"], "url": ""}
            )
        for r in data.get("results", [])[:_MAX_RESULTS]:
            results.append(
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("content", ""),
                    "url": r.get("url", ""),
                }
            )
        return ToolResult(tool_name=self.name, success=True, data=results)

    # ── Serper ────────────────────────────────────────────────────────────────

    async def _serper(self, query: str) -> ToolResult:
        url = "https://google.serper.dev/search"
        headers = {
            **_BASE_HEADERS,
            "X-API-KEY": self._serper_key,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": _MAX_RESULTS}
        timeout = aiohttp.ClientTimeout(total=_TIMEOUT_S)
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                resp.raise_for_status()
                data = await resp.json()

        results: list[dict] = []
        ab = data.get("answerBox", {})
        if ab:
            results.append(
                {
                    "title": ab.get("title", "Answer"),
                    "snippet": ab.get("answer") or ab.get("snippet", ""),
                    "url": ab.get("link", ""),
                }
            )
        for r in data.get("organic", [])[:_MAX_RESULTS]:
            results.append(
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("snippet", ""),
                    "url": r.get("link", ""),
                }
            )
        return ToolResult(tool_name=self.name, success=True, data=results)

    # ── Brave Search ──────────────────────────────────────────────────────────

    async def _brave(self, query: str) -> ToolResult:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            **_BASE_HEADERS,
            "X-Subscription-Token": self._brave_key,
        }
        params = {"q": query, "count": _MAX_RESULTS, "text_decorations": "false"}
        timeout = aiohttp.ClientTimeout(total=_TIMEOUT_S)
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, params=params, timeout=timeout) as resp:
                resp.raise_for_status()
                data = await resp.json()

        results: list[dict] = []
        for r in data.get("web", {}).get("results", [])[:_MAX_RESULTS]:
            results.append(
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("description", ""),
                    "url": r.get("url", ""),
                }
            )
        return ToolResult(tool_name=self.name, success=True, data=results)

    # ── DuckDuckGo (no key) ───────────────────────────────────────────────────

    async def _duckduckgo(self, query: str) -> ToolResult:
        """
        Two-stage DDG search:
          1. Instant Answer API (structured, fast, often empty for current events)
          2. duckduckgo-search library (HTML scraping, broader coverage)
        """
        result = await self._ddg_instant(query)
        if result.success and result.data:
            return result

        # Fall through to library-based scraping
        return await self._ddg_library(query)

    async def _ddg_instant(self, query: str) -> ToolResult:
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
        }
        timeout = aiohttp.ClientTimeout(total=_TIMEOUT_S)
        try:
            async with aiohttp.ClientSession(headers=_BASE_HEADERS) as session:
                async with session.get(url, params=params, timeout=timeout) as resp:
                    resp.raise_for_status()
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.debug("DDG instant answer failed: %s", exc)
            return ToolResult(tool_name=self.name, success=False, data=[], error=str(exc))

        results: list[dict] = []
        if data.get("AbstractText"):
            results.append(
                {
                    "title": data.get("Heading", ""),
                    "snippet": data["AbstractText"],
                    "url": data.get("AbstractURL", ""),
                }
            )
        for topic in data.get("RelatedTopics", [])[:_MAX_RESULTS]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(
                    {
                        "title": topic.get("Text", "")[:80],
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                    }
                )

        return ToolResult(tool_name=self.name, success=bool(results), data=results)

    async def _ddg_library(self, query: str) -> ToolResult:
        """Use the duckduckgo-search library if installed."""
        try:
            import asyncio
            from duckduckgo_search import DDGS  # type: ignore[import]

            loop = asyncio.get_running_loop()

            def _sync_search() -> list[dict]:
                with DDGS() as ddgs:
                    return [
                        {
                            "title": r.get("title", ""),
                            "snippet": r.get("body", ""),
                            "url": r.get("href", ""),
                        }
                        for r in ddgs.text(query, max_results=_MAX_RESULTS)
                    ]

            results = await loop.run_in_executor(None, _sync_search)
            if results:
                return ToolResult(tool_name=self.name, success=True, data=results)
        except ImportError:
            logger.debug(
                "duckduckgo-search not installed; "
                "run `pip install duckduckgo-search` for better DDG coverage"
            )
        except Exception as exc:
            logger.warning("DDG library search failed: %s", exc)

        return ToolResult(
            tool_name=self.name,
            success=False,
            data=None,
            error=(
                "All search backends unavailable. "
                "Set TAVILY_API_KEY, SERPER_API_KEY, or BRAVE_API_KEY for "
                "reliable search, or install duckduckgo-search for the free fallback."
            ),
        )
