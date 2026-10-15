"""Built-in tools for real-time weather forecasts, categorized news feeds, and financial quotes.

Provides:
- weather.forecast: Live weather conditions and multi-day forecasts via Open-Meteo.
- news.headlines: Categorized global, tech, business, and science news headlines via RSS feeds.
- finance.quote: Real-time cryptocurrency and stock market price quotes and 24h metrics.
"""

from __future__ import annotations

import asyncio
import html
import json
import re
import time
import urllib.parse
from typing import TYPE_CHECKING, Any

import httpx

from robot.telemetry import get_logger
from robot.tools import PermissionLevel, ToolDefinition

if TYPE_CHECKING:
    from robot.core.engine import RobotEngine

logger = get_logger("tools.weather_news")

# In-memory LRU caches with TTL
_WEATHER_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_NEWS_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_FINANCE_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}

# WMO Weather interpretation codes (WW)
WMO_WEATHER_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    62: "Moderate rain",
    63: "Heavy rain",
    65: "Violent rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

# RSS Feed sources
NEWS_FEEDS: dict[str, list[str]] = {
    "top": [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    ],
    "tech": [
        "https://news.ycombinator.com/rss",
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://www.theverge.com/rss/index.xml",
    ],
    "business": [
        "https://finance.yahoo.com/news/rssindex",
        "https://feeds.bloomberg.com/markets/news.rss",
    ],
    "world": [
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    ],
    "science": [
        "https://www.sciencedaily.com/rss/top/science.xml",
        "http://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    ],
}

# Crypto ID mapping for CoinGecko
CRYPTO_ALIASES: dict[str, str] = {
    "btc": "bitcoin",
    "bitcoin": "bitcoin",
    "eth": "ethereum",
    "ethereum": "ethereum",
    "sol": "solana",
    "solana": "solana",
    "doge": "dogecoin",
    "dogecoin": "dogecoin",
    "ada": "cardano",
    "cardano": "cardano",
    "xrp": "ripple",
    "ripple": "ripple",
    "bnb": "binancecoin",
    "binance": "binancecoin",
    "dot": "polkadot",
    "polkadot": "polkadot",
    "matic": "matic-network",
    "polygon": "matic-network",
    "avax": "avalanche-2",
    "avalanche": "avalanche-2",
}

STOCK_ALIASES: dict[str, str] = {
    "apple": "AAPL",
    "aapl": "AAPL",
    "google": "GOOGL",
    "googl": "GOOGL",
    "alphabet": "GOOGL",
    "microsoft": "MSFT",
    "msft": "MSFT",
    "nvidia": "NVDA",
    "nvda": "NVDA",
    "tesla": "TSLA",
    "tsla": "TSLA",
    "amazon": "AMZN",
    "amzn": "AMZN",
    "meta": "META",
    "facebook": "META",
    "sp500": "^GSPC",
    "s&p 500": "^GSPC",
    "s&p": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "dow jones": "^DJI",
}


async def fetch_weather(location: str = "Pune") -> dict[str, Any]:
    """Fetch live weather and 3-day forecast using Open-Meteo APIs.

    Args:
        location: City, locality, or country name (e.g. 'Pune', 'Tokyo', 'New York').

    Returns:
        Structured weather data dictionary with temperatures, humidity, wind, and forecast.
    """
    clean_loc = (location or "Pune").strip().title()
    now_ts = time.time()

    # Check cache (15 min TTL)
    if clean_loc.lower() in _WEATHER_CACHE:
        cached_time, cached_data = _WEATHER_CACHE[clean_loc.lower()]
        if now_ts - cached_time < 900:
            return cached_data

    headers = {"User-Agent": "RobotCore/2.0 (Open-Source AI Assistant)"}

    try:
        async with httpx.AsyncClient(headers=headers, timeout=4.0) as client:
            # 1. Geocode location to lat/lon
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(clean_loc)}&count=1&language=en&format=json"
            geo_resp = await client.get(geo_url)
            if geo_resp.status_code != 200:
                return {"status": "error", "message": f"Could not geocode location '{clean_loc}'."}

            geo_data = geo_resp.json()
            results = geo_data.get("results", [])
            if not results:
                return {"status": "not_found", "message": f"Location '{clean_loc}' was not found in global geocoding database."}

            loc_entry = results[0]
            lat = loc_entry["latitude"]
            lon = loc_entry["longitude"]
            display_name = f"{loc_entry.get('name')}, {loc_entry.get('admin1', '')}, {loc_entry.get('country', '')}".replace(", ,", ",").strip(" ,")

            # 2. Fetch current weather and daily forecast
            forecast_url = (
                f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"
                f"&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
                f"&timezone=auto"
            )
            fc_resp = await client.get(forecast_url)
            if fc_resp.status_code != 200:
                return {"status": "error", "message": f"Failed to retrieve weather data for '{display_name}'."}

            fc_data = fc_resp.json()
            curr = fc_data.get("current", {})
            daily = fc_data.get("daily", {})

            w_code = curr.get("weather_code", 0)
            condition_desc = WMO_WEATHER_CODES.get(w_code, "Fair")
            temp_c = curr.get("temperature_2m", 0.0)
            feels_like_c = curr.get("apparent_temperature", temp_c)
            humidity = curr.get("relative_humidity_2m", 0)
            wind_kmh = curr.get("wind_speed_10m", 0.0)
            precip = curr.get("precipitation", 0.0)

            daily_forecast = []
            days = daily.get("time", [])
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            day_codes = daily.get("weather_code", [])
            precip_prob = daily.get("precipitation_probability_max", [])

            for i in range(min(3, len(days))):
                d_code = day_codes[i] if i < len(day_codes) else 0
                daily_forecast.append({
                    "date": days[i],
                    "condition": WMO_WEATHER_CODES.get(d_code, "Fair"),
                    "max_temp_c": max_temps[i] if i < len(max_temps) else temp_c,
                    "min_temp_c": min_temps[i] if i < len(min_temps) else temp_c,
                    "precip_probability": f"{precip_prob[i]}%" if i < len(precip_prob) else "0%",
                })

            summary = (
                f"Weather in {display_name}: {condition_desc}, {temp_c}°C (feels like {feels_like_c}°C). "
                f"Humidity: {humidity}%, Wind: {wind_kmh} km/h."
            )

            result = {
                "status": "success",
                "location": display_name,
                "temperature_c": temp_c,
                "feels_like_c": feels_like_c,
                "condition": condition_desc,
                "humidity_percent": humidity,
                "wind_kmh": wind_kmh,
                "precipitation_mm": precip,
                "forecast_3day": daily_forecast,
                "summary": summary,
            }
            _WEATHER_CACHE[clean_loc.lower()] = (now_ts, result)
            return result

    except Exception as e:
        logger.error("weather_fetch_error", location=clean_loc, error=str(e))
        return {"status": "error", "message": f"Weather lookup failed: {e}"}


async def fetch_news_headlines(category: str = "top", limit: int = 5) -> list[dict[str, Any]]:
    """Fetch current news headlines parsed from verified public RSS channels.

    Args:
        category: News category ('top', 'tech', 'business', 'world', 'science').
        limit: Max headlines to return.

    Returns:
        List of news items with title, snippet, source, and link.
    """
    clean_cat = category.strip().lower()
    if clean_cat not in NEWS_FEEDS:
        clean_cat = "top"

    now_ts = time.time()
    cache_key = f"{clean_cat}_{limit}"
    if cache_key in _NEWS_CACHE:
        cached_time, cached_items = _NEWS_CACHE[cache_key]
        if now_ts - cached_time < 600:
            return cached_items

    urls = NEWS_FEEDS[clean_cat]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    headlines: list[dict[str, Any]] = []

    try:
        async with httpx.AsyncClient(headers=headers, timeout=2.0, follow_redirects=True) as client:
            for url in urls:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        items = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL)
                        for it in items:
                            title_m = re.search(r"<title>(.*?)</title>", it, re.DOTALL)
                            link_m = re.search(r"<link>(.*?)</link>", it, re.DOTALL)
                            desc_m = re.search(r"<description>(.*?)</description>", it, re.DOTALL)

                            raw_title = title_m.group(1).strip() if title_m else ""
                            raw_desc = desc_m.group(1).strip() if desc_m else ""
                            raw_link = link_m.group(1).strip() if link_m else ""

                            # Clean CDATA and HTML entities
                            clean_title = html.unescape(re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", raw_title))
                            clean_title = re.sub(r"<[^>]+>", "", clean_title).strip()
                            clean_desc = html.unescape(re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", raw_desc))
                            clean_desc = re.sub(r"<[^>]+>", "", clean_desc).strip()

                            if clean_title and clean_title not in [h["title"] for h in headlines]:
                                source = "BBC News" if "bbc" in url else ("Hacker News" if "ycombinator" in url else ("Ars Technica" if "arstechnica" in url else "Global News"))
                                headlines.append({
                                    "title": clean_title,
                                    "snippet": clean_desc[:250] + "..." if len(clean_desc) > 250 else clean_desc,
                                    "source": source,
                                    "url": raw_link,
                                    "category": clean_cat,
                                })
                                if len(headlines) >= limit:
                                    break
                except Exception:
                    continue
                if len(headlines) >= limit:
                    break

        if headlines:
            _NEWS_CACHE[cache_key] = (now_ts, headlines[:limit])
            return headlines[:limit]
        return [{"status": "empty", "message": f"No recent headlines available for category '{clean_cat}'."}]

    except Exception as e:
        logger.error("news_fetch_error", category=clean_cat, error=str(e))
        return [{"status": "error", "message": f"News feed error: {e}"}]


async def fetch_finance_quote(symbol: str = "bitcoin") -> dict[str, Any]:
    """Fetch live stock market or cryptocurrency quote with 24h change metrics.

    Args:
        symbol: Ticker symbol, stock name, or crypto currency (e.g. 'BTC', 'ETH', 'NVDA', 'AAPL', 'TSLA').

    Returns:
        Structured price quote and market metrics.
    """
    clean_sym = symbol.strip().lower()
    now_ts = time.time()

    if clean_sym in _FINANCE_CACHE:
        cached_time, cached_data = _FINANCE_CACHE[clean_sym]
        if now_ts - cached_time < 180:
            return cached_data

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    # 1. Check if it's a known Cryptocurrency
    crypto_id = CRYPTO_ALIASES.get(clean_sym)
    if crypto_id or clean_sym in {"crypto", "bitcoin", "ethereum", "solana"}:
        target_id = crypto_id or "bitcoin"
        try:
            async with httpx.AsyncClient(headers=headers, timeout=2.0) as client:
                url = f"https://api.coingecko.com/api/v3/simple/price?ids={target_id}&vs_currencies=usd,inr&include_24hr_change=true"
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    if target_id in data:
                        coin = data[target_id]
                        usd_price = coin.get("usd", 0.0)
                        inr_price = coin.get("inr", 0.0)
                        change_24h = coin.get("usd_24h_change", 0.0)
                        sign = "+" if change_24h >= 0 else ""
                        summary = f"{target_id.title()} is currently trading at ${usd_price:,.2f} USD (₹{inr_price:,.2f} INR), {sign}{change_24h:.2f}% in 24 hours."

                        res = {
                            "status": "success",
                            "asset_type": "cryptocurrency",
                            "symbol": target_id.upper(),
                            "name": target_id.title(),
                            "price_usd": usd_price,
                            "price_inr": inr_price,
                            "change_24h_percent": round(change_24h, 2),
                            "summary": summary,
                        }
                        _FINANCE_CACHE[clean_sym] = (now_ts, res)
                        return res
        except Exception:
            pass

    # 2. Check Stocks / Indices via Yahoo Finance Open API
    target_ticker = STOCK_ALIASES.get(clean_sym, clean_sym.upper())
    try:
        async with httpx.AsyncClient(headers=headers, timeout=2.0) as client:
            y_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(target_ticker)}?interval=1d&range=1d"
            resp = await client.get(y_url)
            if resp.status_code == 200:
                y_data = resp.json()
                result_list = y_data.get("chart", {}).get("result", [])
                if result_list:
                    meta = result_list[0].get("meta", {})
                    price = meta.get("regularMarketPrice", 0.0)
                    prev_close = meta.get("chartPreviousClose", price)
                    currency = meta.get("currency", "USD")
                    change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0
                    sign = "+" if change_pct >= 0 else ""
                    name = meta.get("shortName") or meta.get("symbol", target_ticker)
                    summary = f"{name} ({target_ticker}) is at {price:,.2f} {currency}, {sign}{change_pct:.2f}% today."

                    res = {
                        "status": "success",
                        "asset_type": "stock_or_index",
                        "symbol": target_ticker,
                        "name": name,
                        "price": price,
                        "currency": currency,
                        "change_percent": round(change_pct, 2),
                        "summary": summary,
                    }
                    _FINANCE_CACHE[clean_sym] = (now_ts, res)
                    return res
    except Exception:
        pass

    return {
        "status": "not_found",
        "message": f"Could not retrieve live price quote for '{symbol}'.",
    }


def register_weather_news_tools(engine: "RobotEngine") -> None:
    """Register weather, news, and financial quote tools with the engine's tool registry.

    Args:
        engine: Reference to the RobotEngine instance.
    """

    # 1. weather.forecast
    async def weather_forecast(location: str = "Pune", **kwargs: Any) -> dict[str, Any]:
        """Get live weather and multi-day forecast for a city or country."""
        eff_loc = (location or kwargs.get("city") or kwargs.get("place") or kwargs.get("query") or "Pune").strip()
        return await fetch_weather(eff_loc)

    engine.tool_registry.register(
        ToolDefinition(
            name="weather.forecast",
            description="Get real-time live weather conditions (temperature, humidity, wind, condition) and 3-day forecast for any city or location. Spoken triggers: 'What is the weather in Pune?', 'Is it going to rain today?', 'Weather forecast'.",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City or location name (e.g. 'Pune', 'Tokyo', 'London', 'New York')",
                        "default": "Pune",
                    },
                },
            },
            handler=weather_forecast,
            permission_level=PermissionLevel.READ_ONLY,
            family="weather",
            timeout_s=3.0,
            is_idempotent=True,
            max_retries=1,
        )
    )

    # 2. news.headlines
    async def news_headlines(category: str = "top", limit: int = 5, **kwargs: Any) -> dict[str, Any]:
        """Get the latest verified news headlines across global, tech, business, and science feeds."""
        eff_cat = (category or kwargs.get("topic") or kwargs.get("section") or "top").strip().lower()
        eff_limit = limit or kwargs.get("count") or 5
        items = await fetch_news_headlines(category=eff_cat, limit=int(eff_limit))
        return {
            "status": "success",
            "category": eff_cat,
            "count": len(items),
            "headlines": items,
        }

    engine.tool_registry.register(
        ToolDefinition(
            name="news.headlines",
            description="Fetch the latest breaking news headlines across top stories, technology, business, world, or science. Spoken triggers: 'What are the top news headlines?', 'Tech news today', 'Latest world news'.",
            parameters={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "News category: 'top', 'tech', 'business', 'world', or 'science'",
                        "default": "top",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of headlines to return (1-10)",
                        "default": 5,
                    },
                },
            },
            handler=news_headlines,
            permission_level=PermissionLevel.READ_ONLY,
            family="news",
            timeout_s=3.0,
            is_idempotent=True,
            max_retries=1,
        )
    )

    # 3. finance.quote
    async def finance_quote(symbol: str = "bitcoin", **kwargs: Any) -> dict[str, Any]:
        """Get live cryptocurrency or stock market price quote."""
        eff_sym = (symbol or kwargs.get("ticker") or kwargs.get("coin") or kwargs.get("stock") or kwargs.get("asset") or "bitcoin").strip()
        return await fetch_finance_quote(eff_sym)

    engine.tool_registry.register(
        ToolDefinition(
            name="finance.quote",
            description="Get real-time market prices and 24-hour percentage changes for cryptocurrencies (Bitcoin, Ethereum, Solana, Doge) and stocks/indices (NVIDIA, Apple, Tesla, S&P 500). Spoken triggers: 'What is the Bitcoin price right now?', 'Check NVIDIA stock', 'How is Tesla doing today?'.",
            parameters={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Crypto or stock ticker/name (e.g. 'bitcoin', 'btc', 'ethereum', 'nvda', 'aapl', 'tsla', 'sp500')",
                    },
                },
                "required": ["symbol"],
            },
            handler=finance_quote,
            permission_level=PermissionLevel.READ_ONLY,
            family="finance",
            timeout_s=3.0,
            is_idempotent=True,
            max_retries=1,
        )
    )
