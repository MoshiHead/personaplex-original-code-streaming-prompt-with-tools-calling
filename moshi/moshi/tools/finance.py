"""
Finance tool for PersonaPlex — stocks, commodities, forex, and market indices.

Primary data source: yfinance (Yahoo Finance), which requires no API key.
Install with: pip install yfinance

Falls back to a direct Yahoo Finance quote API call if yfinance is not installed.

Covers:
  - Individual stocks (Tesla, Apple, Microsoft, Google, Amazon, Nvidia, Meta, …)
  - Commodities (gold, silver, crude oil, Brent, natural gas, platinum)
  - Forex pairs (USD/BDT, USD/INR, EUR/USD, GBP/USD, …)
  - Market indices (S&P 500, Nasdaq, Dow Jones, FTSE, Nikkei, Hang Seng, DAX)
  - Explicit ticker symbols (e.g. "$TSLA", "AAPL")
"""

import asyncio
import logging
import re
import urllib.parse
from typing import Optional

import aiohttp

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_TIMEOUT_S = 12

# ──────────────────────────────────────────────────────────────────────────────
# Symbol lookup tables
# ──────────────────────────────────────────────────────────────────────────────

_STOCKS: dict[str, tuple[str, str]] = {
    # keyword → (display_name, yahoo_ticker)
    "tesla": ("Tesla", "TSLA"),
    "tsla": ("Tesla", "TSLA"),
    "apple": ("Apple", "AAPL"),
    "aapl": ("Apple", "AAPL"),
    "microsoft": ("Microsoft", "MSFT"),
    "msft": ("Microsoft", "MSFT"),
    "google": ("Alphabet / Google", "GOOGL"),
    "googl": ("Alphabet / Google", "GOOGL"),
    "alphabet": ("Alphabet / Google", "GOOGL"),
    "amazon": ("Amazon", "AMZN"),
    "amzn": ("Amazon", "AMZN"),
    "nvidia": ("Nvidia", "NVDA"),
    "nvda": ("Nvidia", "NVDA"),
    "meta": ("Meta", "META"),
    "facebook": ("Meta", "META"),
    "netflix": ("Netflix", "NFLX"),
    "nflx": ("Netflix", "NFLX"),
    "jpmorgan": ("JPMorgan Chase", "JPM"),
    "jpm": ("JPMorgan Chase", "JPM"),
    "bank of america": ("Bank of America", "BAC"),
    "bac": ("Bank of America", "BAC"),
    "goldman sachs": ("Goldman Sachs", "GS"),
    "berkshire": ("Berkshire Hathaway", "BRK-B"),
    "palantir": ("Palantir", "PLTR"),
    "amd": ("AMD", "AMD"),
    "intel": ("Intel", "INTC"),
    "intc": ("Intel", "INTC"),
    "samsung": ("Samsung", "005930.KS"),
    "alibaba": ("Alibaba", "BABA"),
    "baba": ("Alibaba", "BABA"),
    "spotify": ("Spotify", "SPOT"),
    "uber": ("Uber", "UBER"),
    "airbnb": ("Airbnb", "ABNB"),
    "coinbase": ("Coinbase", "COIN"),
    "openai": ("Microsoft (OpenAI partner)", "MSFT"),  # OpenAI is private
}

_COMMODITIES: dict[str, tuple[str, str]] = {
    "gold": ("Gold", "GC=F"),
    "silver": ("Silver", "SI=F"),
    "oil": ("Crude Oil (WTI)", "CL=F"),
    "crude oil": ("Crude Oil (WTI)", "CL=F"),
    "crude": ("Crude Oil (WTI)", "CL=F"),
    "brent": ("Brent Crude Oil", "BZ=F"),
    "natural gas": ("Natural Gas", "NG=F"),
    "platinum": ("Platinum", "PL=F"),
    "copper": ("Copper", "HG=F"),
    "wheat": ("Wheat", "ZW=F"),
    "corn": ("Corn", "ZC=F"),
}

_FOREX: dict[str, tuple[str, str]] = {
    "usd to eur": ("USD/EUR", "USDEUR=X"),
    "usd to gbp": ("USD/GBP", "USDGBP=X"),
    "usd to jpy": ("USD/JPY", "USDJPY=X"),
    "usd to bdt": ("USD/BDT", "USDBDT=X"),
    "usd to inr": ("USD/INR", "USDINR=X"),
    "usd to cad": ("USD/CAD", "USDCAD=X"),
    "usd to aud": ("USD/AUD", "USDAUD=X"),
    "usd to cny": ("USD/CNY", "USDCNY=X"),
    "usd to chf": ("USD/CHF", "USDCHF=X"),
    "usd to myr": ("USD/MYR", "USDMYR=X"),
    "eur to usd": ("EUR/USD", "EURUSD=X"),
    "gbp to usd": ("GBP/USD", "GBPUSD=X"),
    "eur to gbp": ("EUR/GBP", "EURGBP=X"),
    "bdt": ("USD/BDT", "USDBDT=X"),
    "taka": ("USD/BDT", "USDBDT=X"),
    "inr": ("USD/INR", "USDINR=X"),
    "rupee": ("USD/INR", "USDINR=X"),
}

_INDICES: dict[str, tuple[str, str]] = {
    "s&p 500": ("S&P 500", "^GSPC"),
    "s&p500": ("S&P 500", "^GSPC"),
    "s&p": ("S&P 500", "^GSPC"),
    "sp500": ("S&P 500", "^GSPC"),
    "nasdaq": ("Nasdaq Composite", "^IXIC"),
    "dow jones": ("Dow Jones", "^DJI"),
    "dow": ("Dow Jones", "^DJI"),
    "ftse": ("FTSE 100", "^FTSE"),
    "nikkei": ("Nikkei 225", "^N225"),
    "hang seng": ("Hang Seng", "^HSI"),
    "dax": ("DAX", "^GDAXI"),
    "sensex": ("BSE Sensex", "^BSESN"),
    "nifty": ("Nifty 50", "^NSEI"),
    "russell 2000": ("Russell 2000", "^RUT"),
    "vix": ("VIX (Volatility Index)", "^VIX"),
}

# Merged lookup (longer keys first to prevent prefix collisions)
_ALL_LOOKUPS: dict[str, tuple[str, str]] = {
    **_COMMODITIES,
    **_FOREX,
    **_INDICES,
    **_STOCKS,
}
_SORTED_KEYWORDS = sorted(_ALL_LOOKUPS.keys(), key=len, reverse=True)

# Pattern for explicit ticker symbols like $TSLA or AAPL in isolation
_EXPLICIT_TICKER_RE = re.compile(
    r"\$([A-Z]{1,5}(?:[=-][A-Z0-9]+)?)"   # $TSLA
    r"|(?<!\w)([A-Z]{2,5})(?!\w)",          # TSLA (not inside a longer word)
)


class FinanceTool(BaseTool):
    """
    Live financial data: stocks, commodities, forex, and market indices.

    Requires: pip install yfinance
    Falls back to Yahoo Finance direct API if yfinance is unavailable.
    """

    name = "finance"
    description = (
        "Live stock prices, gold/silver/oil prices, currency exchange rates, "
        "and market index values (S&P 500, Nasdaq, Dow, FTSE, Nikkei, …)."
    )

    def __init__(self) -> None:
        try:
            import yfinance  # noqa: F401
            self._has_yfinance = True
            logger.info("FinanceTool initialised — using yfinance")
        except ImportError:
            self._has_yfinance = False
            logger.info(
                "FinanceTool initialised — yfinance not installed, "
                "using Yahoo Finance direct API (install yfinance for better coverage)"
            )

    async def execute(self, query: str, **kwargs) -> ToolResult:
        symbols = self._extract_symbols(query)
        if not symbols:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=None,
                error="No recognisable financial instrument found in the query.",
            )

        if self._has_yfinance:
            return await self._fetch_yfinance(symbols)
        return await self._fetch_yahoo_api(symbols)

    # ── Symbol extraction ─────────────────────────────────────────────────────

    def _extract_symbols(self, query: str) -> list[tuple[str, str]]:
        """Return list of (display_name, yahoo_ticker) from the query."""
        q_lower = query.lower()
        found: list[tuple[str, str]] = []
        seen_tickers: set[str] = set()

        for keyword in _SORTED_KEYWORDS:
            if keyword in q_lower:
                display, ticker = _ALL_LOOKUPS[keyword]
                if ticker not in seen_tickers:
                    found.append((display, ticker))
                    seen_tickers.add(ticker)

        # Also pick up explicit ticker symbols like $TSLA
        for m in _EXPLICIT_TICKER_RE.finditer(query):
            ticker = (m.group(1) or m.group(2)).upper()
            if ticker not in seen_tickers and len(ticker) >= 2:
                # Skip common English words that could match the regex
                if ticker not in {"A", "I", "IT", "OR", "AT", "IS", "TO", "OF", "IN", "ON", "BE", "AS", "WE"}:
                    found.append((ticker, ticker))
                    seen_tickers.add(ticker)

        return found[:8]  # cap at 8 to keep prompt concise

    # ── yfinance path ─────────────────────────────────────────────────────────

    async def _fetch_yfinance(self, symbols: list[tuple[str, str]]) -> ToolResult:
        import yfinance as yf

        loop = asyncio.get_running_loop()
        lines: list[str] = []

        async def _one(display: str, ticker: str) -> Optional[str]:
            try:
                t = await loop.run_in_executor(None, yf.Ticker, ticker)
                info: dict = await loop.run_in_executor(None, lambda: t.fast_info)

                # fast_info is a lightweight dict-like object
                price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
                if price is None:
                    # Fallback: grab last close from history
                    hist = await loop.run_in_executor(
                        None, lambda: t.history(period="1d", auto_adjust=True)
                    )
                    if not hist.empty:
                        price = float(hist["Close"].iloc[-1])

                if price is None:
                    return f"{display} ({ticker}): price unavailable"

                currency = getattr(info, "currency", "USD") or "USD"
                prev_close = getattr(info, "previous_close", None)
                change_pct: Optional[float] = None
                if prev_close and prev_close > 0 and price:
                    change_pct = (price - prev_close) / prev_close * 100

                line = _format_price_line(display, ticker, price, currency, change_pct)
                return line
            except Exception as exc:
                logger.warning("yfinance: %s (%s) — %s", display, ticker, exc)
                return None

        tasks = [_one(d, t) for d, t in symbols]
        results = await asyncio.gather(*tasks)
        lines = [r for r in results if r]

        if not lines:
            # yfinance failed for all — try direct API
            return await self._fetch_yahoo_api(symbols)

        return ToolResult(tool_name=self.name, success=True, data="\n".join(lines))

    # ── Yahoo Finance direct API fallback ─────────────────────────────────────

    async def _fetch_yahoo_api(self, symbols: list[tuple[str, str]]) -> ToolResult:
        lines: list[str] = []
        timeout = aiohttp.ClientTimeout(total=_TIMEOUT_S)

        async def _one(display: str, ticker: str) -> Optional[str]:
            encoded = urllib.parse.quote(ticker)
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
            params = {"interval": "1d", "range": "1d"}
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (PersonaPlex; +https://github.com/NVIDIA/personaplex)"
                )
            }
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(url, params=params, timeout=timeout) as resp:
                        if resp.status != 200:
                            return None
                        data = await resp.json()

                result = data.get("chart", {}).get("result") or []
                if not result:
                    return None
                meta = result[0].get("meta", {})
                price: Optional[float] = (
                    meta.get("regularMarketPrice")
                    or meta.get("chartPreviousClose")
                )
                if price is None:
                    return None
                currency = meta.get("currency", "USD")
                prev_close = meta.get("chartPreviousClose")
                change_pct: Optional[float] = None
                if prev_close and prev_close > 0 and price:
                    change_pct = (price - prev_close) / prev_close * 100

                return _format_price_line(display, ticker, price, currency, change_pct)
            except Exception as exc:
                logger.warning("Yahoo API: %s (%s) — %s", display, ticker, exc)
                return None

        tasks = [_one(d, t) for d, t in symbols]
        results = await asyncio.gather(*tasks)
        lines = [r for r in results if r]

        if not lines:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=None,
                error="Yahoo Finance returned no data for the requested symbols.",
            )
        return ToolResult(tool_name=self.name, success=True, data="\n".join(lines))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_price_line(
    display: str,
    ticker: str,
    price: float,
    currency: str,
    change_pct: Optional[float],
) -> str:
    if price >= 1000:
        price_str = f"{price:,.2f}"
    elif price >= 1:
        price_str = f"{price:.2f}"
    elif price >= 0.01:
        price_str = f"{price:.4f}"
    else:
        price_str = f"{price:.6f}"

    line = f"{display} ({ticker}): {currency} {price_str}"
    if change_pct is not None:
        sign = "+" if change_pct >= 0 else ""
        line += f" ({sign}{change_pct:.2f}% today)"
    return line
