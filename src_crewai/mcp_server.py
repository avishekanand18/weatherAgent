"""
FastMCP server exposing the agri-weather tools over stdio.

Run standalone:
    python src/mcp_server.py
"""
from __future__ import annotations

import requests
from fastmcp import FastMCP
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

mcp = FastMCP("weather-tools")

# Shared session with retry-friendly defaults.
_SESSION = requests.Session()
_TIMEOUT = 15  # seconds

_RETRY = dict(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(
        (requests.exceptions.ConnectionError, requests.exceptions.Timeout)
    ),
)


@retry(**_RETRY)
def _http_get(url: str, *, params: dict, headers: dict | None = None, verify: bool = False) -> requests.Response:
    response = _SESSION.get(url, params=params, headers=headers, timeout=_TIMEOUT, verify=verify)
    response.raise_for_status()
    return response


@mcp.tool()
def get_forecast_for_location(location_name: str) -> str:
    """
    One-shot tool: geocodes a location name and returns its 14-day agricultural
    weather forecast in a compact, LLM-friendly summary.

    Input Example:
        "Bordeaux, France" or "Napa Valley, California"

    Output Example:
        "14-Day Agricultural Forecast for Bordeaux, France (Lat: 44.84, Lon: -0.58):
         Date 2026-05-08: Max Temp 25.5°C, Min Temp 12.0°C, Rain 0.0mm, EvapoT 4.2mm
         ..."
    """
    coords = _geocode(location_name)
    if isinstance(coords, str):
        # Error message bubbles up to the LLM as-is.
        return coords
    lat, lon = coords
    return _fetch_weather(location_name, lat, lon)


# ----------------------------------------------------------------------------
# Internal helpers (deterministic, not exposed as MCP tools).
# ----------------------------------------------------------------------------

def _geocode(location_name: str) -> tuple[str, str] | str:
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": "AgriWeatherForecaster/1.0 (local_dev)"}
    params = {"q": location_name, "format": "json", "limit": 1}

    try:
        response = _http_get(url, params=params, headers=headers, verify=False)
        data = response.json()
        if not data:
            return (
                "Error: Could not find coordinates for this location. "
                "Please try a broader region."
            )
        return str(data[0].get("lat")), str(data[0].get("lon"))
    except requests.exceptions.RequestException as e:
        return f"API Error occurred during geocoding: {e}"


def _fetch_weather(location_name: str, lat: str, lon: str) -> str:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "et0_fao_evapotranspiration",
        ],
        "timezone": "auto",
        "forecast_days": 14,
    }

    try:
        response = _http_get(url, params=params, verify=False)
        data = response.json()
    except requests.exceptions.RequestException as e:
        return f"API Error occurred during weather fetch: {e}"

    daily = data.get("daily", {}) or {}
    dates = daily.get("time", []) or []
    max_temps = daily.get("temperature_2m_max", []) or []
    min_temps = daily.get("temperature_2m_min", []) or []
    rain = daily.get("precipitation_sum", []) or []
    et0 = daily.get("et0_fao_evapotranspiration", []) or []

    if not dates:
        return f"Error: Weather API returned no daily data for Lat: {lat}, Lon: {lon}."

    lines = [
        f"14-Day Agricultural Forecast for {location_name} (Lat: {lat}, Lon: {lon}):"
    ]
    # zip() truncates to the shortest array, so a partial response can't IndexError.
    for d, tmax, tmin, r, e in zip(dates, max_temps, min_temps, rain, et0):
        lines.append(
            f"Date {d}: Max Temp {tmax}°C, Min Temp {tmin}°C, "
            f"Rain {r}mm, EvapoT {e}mm"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    # Default transport is stdio.
    mcp.run()
