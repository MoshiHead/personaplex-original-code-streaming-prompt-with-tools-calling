"""
Cryptocurrency price tool for PersonaPlex.

Uses the CoinGecko public REST API — no API key required for the free tier.
Set COINGECKO_API_KEY to lift rate limits (CoinGecko Pro or Demo key).

Covers all major coins by symbol or common name.  When the query mentions a
specific coin, only that coin is fetched.  When the query is generic ("crypto
price", "cryptocurrency market"), the top coins by market cap are returned.
"""

import logging
import os
from typing import Optional

import aiohttp

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_COINGECKO_BASE = "https://api.coingecko.com/api/v3"
_TIMEOUT_S = 10

# Mapping of common names / symbols → CoinGecko coin IDs
_SYMBOL_TO_ID: dict[str, str] = {
    # Bitcoin
    "btc": "bitcoin", "bitcoin": "bitcoin",
    # Ethereum
    "eth": "ethereum", "ethereum": "ethereum",
    # Solana
    "sol": "solana", "solana": "solana",
    # BNB / Binance
    "bnb": "binancecoin", "binance coin": "binancecoin", "binance": "binancecoin",
    # Cardano
    "ada": "cardano", "cardano": "cardano",
    # XRP / Ripple
    "xrp": "ripple", "ripple": "ripple",
    # Dogecoin
    "doge": "dogecoin", "dogecoin": "dogecoin",
    # Polkadot
    "dot": "polkadot", "polkadot": "polkadot",
    # Avalanche
    "avax": "avalanche-2", "avalanche": "avalanche-2",
    # Chainlink
    "link": "chainlink", "chainlink": "chainlink",
    # Polygon / Matic
    "matic": "matic-network", "polygon": "matic-network",
    # Litecoin
    "ltc": "litecoin", "litecoin": "litecoin",
    # Shiba Inu
    "shib": "shiba-inu", "shiba": "shiba-inu", "shiba inu": "shiba-inu",
    # Tether
    "usdt": "tether", "tether": "tether",
    # USD Coin
    "usdc": "usd-coin", "usd coin": "usd-coin",
    # Toncoin
    "ton": "the-open-network", "toncoin": "the-open-network",
    # NEAR
    "near": "near",
    # Cosmos
    "atom": "cosmos", "cosmos": "cosmos",
    # Stellar
    "xlm": "stellar", "stellar": "stellar",
    # Uniswap
    "uni": "uniswap", "uniswap": "uniswap",
    # Aave
    "aave": "aave",
    # Ethereum Classic
    "etc": "ethereum-classic",
    # Filecoin
    "fil": "filecoin",
    # Internet Computer
    "icp": "internet-computer",
    # VeChain
    "vet": "vechain", "vechain": "vechain",
    # Monero
    "xmr": "monero", "monero": "monero",
    # Algorand
    "algo": "algorand", "algorand": "algorand",
    # Hedera
    "hbar": "hedera-hashgraph", "hedera": "hedera-hashgraph",
    # Aptos
    "apt": "aptos", "aptos": "aptos",
    # Arbitrum
    "arb": "arbitrum",
    # Optimism
    "op": "optimism",
    # Sui
    "sui": "sui",
}

# Default coins to return when no specific coin is requested
_DEFAULT_COIN_IDS = [
    "bitcoin", "ethereum", "solana", "binancecoin", "ripple",
]


class CryptoTool(BaseTool):
    """
    Live cryptocurrency price data via CoinGecko.

    Environment variables:
        COINGECKO_API_KEY — optional; lifts rate limits on Pro/Demo tier
    """

    name = "crypto"
    description = (
        "Live cryptocurrency prices and 24-hour change from CoinGecko. "
        "Covers Bitcoin, Ethereum, Solana, BNB, and 20+ major coins."
    )

    def __init__(self) -> None:
        self._api_key = os.getenv("COINGECKO_API_KEY", "").strip()
        tier = "Pro/Demo" if self._api_key else "free (no key)"
        logger.info("CryptoTool initialised — CoinGecko %s tier", tier)

    async def execute(self, query: str, **kwargs) -> ToolResult:
        coin_ids = self._extract_coin_ids(query)
        if not coin_ids:
            coin_ids = _DEFAULT_COIN_IDS
            logger.debug("no specific coin detected — fetching top coins")

        return await self._fetch_prices(coin_ids)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_coin_ids(self, query: str) -> list[str]:
        """Extract CoinGecko IDs from the query by matching names / symbols."""
        q = query.lower()
        found: list[str] = []
        seen: set[str] = set()

        # Sort by keyword length descending so "ethereum classic" matches before "ethereum"
        for keyword in sorted(_SYMBOL_TO_ID, key=len, reverse=True):
            if keyword in q:
                coin_id = _SYMBOL_TO_ID[keyword]
                if coin_id not in seen:
                    found.append(coin_id)
                    seen.add(coin_id)

        return found

    async def _fetch_prices(self, coin_ids: list[str]) -> ToolResult:
        ids_param = ",".join(coin_ids[:12])  # CoinGecko allows many per call
        url = f"{_COINGECKO_BASE}/simple/price"
        params = {
            "ids": ids_param,
            "vs_currencies": "usd",
            "include_24hr_change": "true",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
        }
        headers: dict[str, str] = {"Accept": "application/json"}
        if self._api_key:
            # CoinGecko Demo key goes in x-cg-demo-api-key, Pro in x-cg-pro-api-key
            headers["x-cg-demo-api-key"] = self._api_key

        timeout = aiohttp.ClientTimeout(total=_TIMEOUT_S)
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, params=params, timeout=timeout) as resp:
                if resp.status == 429:
                    return ToolResult(
                        tool_name=self.name,
                        success=False,
                        data=None,
                        error=(
                            "CoinGecko rate limit hit. "
                            "Set COINGECKO_API_KEY to lift limits."
                        ),
                    )
                resp.raise_for_status()
                data: dict = await resp.json()

        if not data:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=None,
                error="CoinGecko returned no price data.",
            )

        lines: list[str] = []
        for coin_id, info in data.items():
            price: Optional[float] = info.get("usd")
            change: Optional[float] = info.get("usd_24h_change")
            market_cap: Optional[float] = info.get("usd_market_cap")

            name = _coin_display_name(coin_id)

            if price is None:
                continue

            # Format price sensibly (small coins can be <$1)
            if price >= 1:
                price_str = f"${price:,.2f}"
            elif price >= 0.01:
                price_str = f"${price:.4f}"
            else:
                price_str = f"${price:.8f}"

            line = f"{name}: {price_str} USD"

            if change is not None:
                sign = "+" if change >= 0 else ""
                line += f" ({sign}{change:.1f}% in 24h)"

            if market_cap and market_cap > 1e6:
                line += f", market cap ${market_cap / 1e9:.1f}B" if market_cap > 1e9 else f", market cap ${market_cap / 1e6:.0f}M"

            lines.append(line)

        if not lines:
            return ToolResult(
                tool_name=self.name,
                success=False,
                data=None,
                error="CoinGecko returned data but no valid prices could be parsed.",
            )

        return ToolResult(tool_name=self.name, success=True, data="\n".join(lines))


# ── Module-level helpers ──────────────────────────────────────────────────────

def _coin_display_name(coin_id: str) -> str:
    """Convert a CoinGecko ID to a human-readable display name."""
    _overrides = {
        "bitcoin": "Bitcoin (BTC)",
        "ethereum": "Ethereum (ETH)",
        "solana": "Solana (SOL)",
        "binancecoin": "BNB (BNB)",
        "ripple": "XRP (XRP)",
        "dogecoin": "Dogecoin (DOGE)",
        "cardano": "Cardano (ADA)",
        "polkadot": "Polkadot (DOT)",
        "avalanche-2": "Avalanche (AVAX)",
        "chainlink": "Chainlink (LINK)",
        "matic-network": "Polygon (MATIC)",
        "litecoin": "Litecoin (LTC)",
        "shiba-inu": "Shiba Inu (SHIB)",
        "tether": "Tether (USDT)",
        "usd-coin": "USD Coin (USDC)",
        "the-open-network": "Toncoin (TON)",
    }
    return _overrides.get(coin_id, coin_id.replace("-", " ").title())
