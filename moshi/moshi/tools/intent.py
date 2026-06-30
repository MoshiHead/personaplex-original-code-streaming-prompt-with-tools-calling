"""
Keyword-based live-data intent detection for PersonaPlex tool routing.

No extra LLM call, no added latency — pure regex matching against the user's
typed query. Returns the categories of live data required so ToolManager can
select the right tools.

Design rationale:
  The full-duplex S2S pipeline has no ASR output to analyze. The only text
  we have is the user's typed "text_prompt" field in the web UI.  A fast
  regex scan keeps augment-prompt latency well under the model's own warm-up
  time (~400ms for warmup + system prompt loading).
"""

import re
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Patterns that indicate the query REQUIRES live / real-time data
# ──────────────────────────────────────────────────────────────────────────────

_LIVE_GROUPS: dict[str, str] = {
    # General recency signals
    "recency": (
        r"\b("
        r"today|yesterday|right now|currently|current|live|real.?time"
        r"|latest|recent|breaking|this (week|month|year)|at the moment"
        r"|as of (today|now|this moment)|now|just happened|today's"
        r")\b"
    ),
    # Crypto currencies and DeFi
    "crypto": (
        r"\b("
        r"bitcoin|btc|ethereum|eth|solana|sol|bnb|binance.?coin"
        r"|cardano|ada|xrp|ripple|dogecoin|doge|litecoin|ltc"
        r"|polkadot|dot|avalanche|avax|chainlink|link"
        r"|polygon|matic|shiba|shib|tether|usdt|usdc|usd.?coin"
        r"|ton|near|cosmos|atom|stellar|xlm|vechain|vet"
        r"|crypto(currency)?|coin.?price|token.?price|defi|nft|altcoin"
        r")\b"
    ),
    # Stocks, indices, commodities, and forex
    "finance": (
        r"\b("
        r"stock.?price|share.?price|market.?cap|trading.?at|stock.?market"
        r"|wall.?street|nasdaq|s&p.?500|sp.?500|dow.?jones|dow|ftse|nikkei"
        r"|hang.?seng|dax|sensex|nifty"
        r"|tesla|tsla|apple|aapl|microsoft|msft|google|googl|alphabet"
        r"|amazon|amzn|nvidia|nvda|meta|facebook|netflix|nflx"
        r"|jpmorgan|jpm|goldman|bank.?of.?america|bac"
        r"|gold.?price|silver.?price|oil.?price|crude.?oil|brent|natural.?gas|platinum"
        r"|exchange.?rate|forex|currency.?rate"
        r"|usd.?to|eur.?to|gbp.?to|jpy.?to|bdt|inr|cad|aud|cny|chf"
        r"|to.?(usd|eur|gbp|jpy|bdt|inr)"
        r")\b"
    ),
    # Weather
    "weather": (
        r"\b("
        r"weather|temperature|forecast|rain(ing)?|sunny|cloudy|wind.?speed"
        r"|humidity|feels.?like|celsius|fahrenheit|snow|storm|hurricane"
        r")\b"
    ),
    # News and current events
    "news": (
        r"\b("
        r"news|headlines|breaking.?news|latest.?news|recent.?news"
        r"|what.?happened|what.?('s|is).?happening|current.?events?"
        r")\b"
    ),
    # People, politics, and leadership
    "people": (
        r"\b("
        r"current (president|prime.?minister|ceo|chancellor|leader|ruler|king|queen|governor|secretary)"
        r"|who.?is.?the.?(current|new|latest)"
        r"|who.?is.?(now|currently|today)"
        r"|election.?results?|who.?won|who.?lost|who.?leads?"
        r"|new.?president|new.?prime.?minister"
        r")\b"
    ),
    # Sports results
    "sports": (
        r"\b("
        r"match.?result|game.?result|final.?score|score"
        r"|standings|championship|tournament.?result"
        r"|premier.?league|champions.?league|la.?liga|bundesliga|serie.?a"
        r"|world.?cup|super.?bowl|nba|nfl|mlb|nhl|euro.?cup"
        r")\b"
    ),
    # AI / tech news (named providers the model shouldn't hallucinate about)
    "ai_news": (
        r"\b("
        r"openai|chatgpt|gpt-[0-9]|claude|anthropic|gemini|llama|mistral"
        r"|grok|ai.?news|artificial.?intelligence.?news|latest.?ai"
        r"|new.?model|new.?llm|new.?ai"
        r")\b"
    ),
}

# Compile once at module load
_LIVE_COMPILED: dict[str, re.Pattern] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in _LIVE_GROUPS.items()
}

# ──────────────────────────────────────────────────────────────────────────────
# Patterns that strongly indicate STATIC knowledge (counter-signal)
# ──────────────────────────────────────────────────────────────────────────────

_STATIC_PATTERN = re.compile(
    r"\b("
    r"explain|how.?does|what.?is (a |an |the )?(?!current|latest|recent|new)"
    r"|definition.?of|define|describe"
    r"|history.?of|biography|how.?to|tutorial|example"
    r"|syntax|algorithm|formula|theory|concept|meaning.?of"
    r"|difference.?between|compare|vs\.?"
    r"|write (a|an|some|me|code|an? email|a letter)"
    r"|translate|summarize|paraphrase|create|generate"
    r"|who.?(invented|discovered|created|founded|wrote|made|built)"
    r"|capital.?(of|city)|population|area|geography"
    r")\b",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_live_intent(query: str) -> tuple[bool, list[str]]:
    """
    Determine whether a query requires live data and which tool categories.

    Returns
    -------
    (needs_live_data, categories)
        needs_live_data — True when at least one live pattern matched
        categories      — subset of {"crypto", "finance", "weather",
                          "news", "people", "sports", "ai_news", "search"}
                          "search" is the catch-all when no specific category
                          matched but a recency signal was present
    """
    if not query or not query.strip():
        return False, []

    q = query

    matched_categories: list[str] = []
    has_recency = bool(_LIVE_COMPILED["recency"].search(q))

    for cat, pattern in _LIVE_COMPILED.items():
        if cat == "recency":
            continue
        if pattern.search(q):
            matched_categories.append(cat)

    if not matched_categories and not has_recency:
        return False, []

    # Suppress false positives: if purely static intent and no specific category
    if not matched_categories and _STATIC_PATTERN.search(q):
        return False, []

    # Collapse categories to tool names
    tool_categories: list[str] = []

    if "crypto" in matched_categories:
        tool_categories.append("crypto")

    finance_cats = {"finance", "weather"}
    if finance_cats & set(matched_categories):
        tool_categories.append("finance")

    # Only add "search" for specific searchable topic categories.
    # Do NOT add "search" just because a recency keyword ("real-time", "live") was
    # present without a specific topic — that case is a general "proactive" signal
    # handled separately by is_proactive_trigger().  Calling the search tool with a
    # system-prompt string as the query produces useless results.
    searchable_cats = {"news", "people", "sports", "ai_news", "weather"}
    if searchable_cats & set(matched_categories):
        tool_categories.append("search")

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in tool_categories:
        if c not in seen:
            unique.append(c)
            seen.add(c)

    # If has_recency but no specific topics: signal "proactive" so callers know a
    # comprehensive data snapshot is appropriate (see is_proactive_trigger below).
    if not unique and has_recency:
        return True, ["proactive"]

    return bool(unique), unique


# ──────────────────────────────────────────────────────────────────────────────
# Proactive trigger detection
# ──────────────────────────────────────────────────────────────────────────────

_PROACTIVE_RE = re.compile(
    r"\b("
    r"real.?time|live.?(data|information|info|prices?|market|news)"
    r"|up.?to.?date|current.?information|always.?accurate"
    r"|latest.?information|fresh.?data|live.?assistant"
    r")\b",
    re.IGNORECASE,
)


def is_proactive_trigger(text: str) -> bool:
    """
    Return True when the text contains a general 'real-time / live data' signal
    without naming a specific topic — i.e. it reads like a system prompt for a
    general real-time assistant rather than a user query about a specific subject.

    Used to decide whether to pre-fetch a comprehensive live data snapshot
    (crypto prices + stock indices + key forex) before the conversation starts.
    """
    return bool(_PROACTIVE_RE.search(text))


def describe_intent(query: str) -> str:
    """Human-readable summary of what was detected (for logging / debugging)."""
    needs_live, categories = detect_live_intent(query)
    if not needs_live:
        return "static knowledge — no tools needed"
    return f"live data required — categories: {', '.join(categories)}"
