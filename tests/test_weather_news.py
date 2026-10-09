"""Unit tests for Phase 3 real-time weather, news, and financial feeds tools.

Validates:
- Open-Meteo geocoding and live weather/forecast retrieval with caching.
- Categorized RSS news feed aggregation and parsing.
- Crypto (CoinGecko) and stock/index (Yahoo Finance) market quotes.
- Tool registration, permissions, argument normalization, and error fallbacks.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robot.tools import PermissionLevel, ToolRegistry
from robot.tools.builtins.weather_news import (
    WMO_WEATHER_CODES,
    _FINANCE_CACHE,
    _NEWS_CACHE,
    _WEATHER_CACHE,
    fetch_finance_quote,
    fetch_news_headlines,
    fetch_weather,
    register_weather_news_tools,
)


@pytest.fixture(autouse=True)
def clear_caches() -> None:
    """Clear in-memory caches between unit tests."""
    _WEATHER_CACHE.clear()
    _NEWS_CACHE.clear()
    _FINANCE_CACHE.clear()


@pytest.fixture
def mock_engine() -> MagicMock:
    """Fixture providing a mock RobotEngine with tool registry."""
    engine = MagicMock()
    engine.tool_registry = ToolRegistry()
    engine.personalities = MagicMock()
    engine.personalities.active = MagicMock(memory_namespace="test")
    engine.state = MagicMock()
    engine.telemetry = MagicMock()
    engine.scheduler = MagicMock()
    return engine


def test_wmo_weather_code_constants() -> None:
    """Verify WMO weather code dictionary has expected descriptions."""
    assert WMO_WEATHER_CODES[0] == "Clear sky"
    assert WMO_WEATHER_CODES[61] == "Slight rain"
    assert WMO_WEATHER_CODES[95] == "Thunderstorm"


@pytest.mark.asyncio
async def test_fetch_weather_success() -> None:
    """Verify live weather forecast retrieval via mocked HTTP responses."""
    mock_geo_resp = MagicMock()
    mock_geo_resp.status_code = 200
    mock_geo_resp.json.return_value = {
        "results": [
            {
                "name": "Pune",
                "country": "India",
                "admin1": "Maharashtra",
                "latitude": 18.5204,
                "longitude": 73.8567,
            }
        ]
    }

    mock_fc_resp = MagicMock()
    mock_fc_resp.status_code = 200
    mock_fc_resp.json.return_value = {
        "current": {
            "temperature_2m": 28.5,
            "apparent_temperature": 30.1,
            "relative_humidity_2m": 65,
            "weather_code": 1,
            "wind_speed_10m": 12.4,
            "precipitation": 0.0,
        },
        "daily": {
            "time": ["2026-10-09", "2026-10-10", "2026-10-11"],
            "weather_code": [1, 2, 61],
            "temperature_2m_max": [31.0, 30.5, 27.0],
            "temperature_2m_min": [21.0, 20.0, 19.5],
            "precipitation_probability_max": [10, 20, 85],
        },
    }

    mock_client = AsyncMock()
    mock_client.get.side_effect = [mock_geo_resp, mock_fc_resp]
    mock_client.__aenter__.return_value = mock_client

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await fetch_weather("Pune")

        assert result["status"] == "success"
        assert "Pune" in result["location"]
        assert result["temperature_c"] == 28.5
        assert result["condition"] == "Mainly clear"
        assert len(result["forecast_3day"]) == 3
        assert result["forecast_3day"][2]["condition"] == "Slight rain"


@pytest.mark.asyncio
async def test_fetch_weather_location_not_found() -> None:
    """Verify handling when geocoding returns no results."""
    mock_geo_resp = MagicMock()
    mock_geo_resp.status_code = 200
    mock_geo_resp.json.return_value = {"results": []}

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_geo_resp
    mock_client.__aenter__.return_value = mock_client

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await fetch_weather("NonExistentCityXYZ123")
        assert result["status"] == "not_found"


@pytest.mark.asyncio
async def test_fetch_news_headlines_success() -> None:
    """Verify RSS feed parsing for news headlines."""
    sample_rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
        <channel>
            <title>Top News</title>
            <item>
                <title><![CDATA[Major AI Breakthrough Announced]]></title>
                <link>https://example.com/ai-news</link>
                <description>Scientists reveal autonomous edge robotics system.</description>
            </item>
            <item>
                <title>Global Markets Surge</title>
                <link>https://example.com/markets</link>
                <description>Tech shares lead the rally.</description>
            </item>
        </channel>
    </rss>"""

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = sample_rss

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp
    mock_client.__aenter__.return_value = mock_client

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await fetch_news_headlines("tech", limit=5)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["title"] == "Major AI Breakthrough Announced"
        assert result[0]["url"] == "https://example.com/ai-news"


@pytest.mark.asyncio
async def test_fetch_crypto_quote_coingecko() -> None:
    """Verify CoinGecko crypto quote lookup."""
    mock_cg_resp = MagicMock()
    mock_cg_resp.status_code = 200
    mock_cg_resp.json.return_value = {
        "bitcoin": {
            "usd": 95000.50,
            "inr": 7900000.0,
            "usd_24h_change": 3.45,
        }
    }

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_cg_resp
    mock_client.__aenter__.return_value = mock_client

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await fetch_finance_quote("bitcoin")

        assert result["status"] == "success"
        assert result["asset_type"] == "cryptocurrency"
        assert result["symbol"] == "BITCOIN"
        assert result["price_usd"] == 95000.50
        assert result["change_24h_percent"] == 3.45


@pytest.mark.asyncio
async def test_fetch_stock_quote_yahoo() -> None:
    """Verify Yahoo Finance stock quote lookup."""
    mock_yahoo_resp = MagicMock()
    mock_yahoo_resp.status_code = 200
    mock_yahoo_resp.json.return_value = {
        "chart": {
            "result": [
                {
                    "meta": {
                        "symbol": "NVDA",
                        "shortName": "NVIDIA Corporation",
                        "regularMarketPrice": 145.20,
                        "chartPreviousClose": 140.00,
                        "currency": "USD",
                    }
                }
            ],
            "error": None,
        }
    }

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_yahoo_resp
    mock_client.__aenter__.return_value = mock_client

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await fetch_finance_quote("NVDA")

        assert result["status"] == "success"
        assert result["asset_type"] == "stock_or_index"
        assert result["symbol"] == "NVDA"
        assert result["name"] == "NVIDIA Corporation"
        assert result["price"] == 145.20
        assert result["change_percent"] == 3.71


def test_register_weather_news_tools(mock_engine: MagicMock) -> None:
    """Verify tool registration into registry."""
    register_weather_news_tools(mock_engine)

    registry: ToolRegistry = mock_engine.tool_registry
    assert "weather.forecast" in registry.list_tools()
    assert "news.headlines" in registry.list_tools()
    assert "finance.quote" in registry.list_tools()

    tool = registry.get("weather.forecast")
    assert tool is not None
    assert tool.permission_level == PermissionLevel.READ_ONLY
    assert tool.family == "weather"
