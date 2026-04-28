"""
FastMCP server exposing the agri-weather tools over stdio.

Run standalone:
    python src/mcp_server.py
"""
import requests
from fastmcp import FastMCP

mcp = FastMCP("weather-tools")


@mcp.tool()
def get_coordinates(location_name: str) -> str:
    """
    Useful for finding the exact latitude and longitude of a city, region, or address.
    Always use this tool first when given a location name by the user.

    Input Example:
        "Bordeaux, France" or "Napa Valley, California"

    Output Example:
        "44.837789,-0.57918" (Returns a comma-separated string of latitude and longitude)
    """
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        "User-Agent": "AgriWeatherForecaster/1.0 (local_dev)"
    }
    params = {
        "q": location_name,
        "format": "json",
        "limit": 1
    }

    try:
        response = requests.get(url, headers=headers, params=params, verify=False)
        response.raise_for_status()
        data = response.json()

        if data:
            lat = data[0].get('lat')
            lon = data[0].get('lon')
            return f"{lat},{lon}"
        else:
            return "Error: Could not find coordinates for this location. Please try a broader region."

    except requests.exceptions.RequestException as e:
        return f"API Error occurred during geocoding: {str(e)}"


@mcp.tool()
def get_agri_weather(coordinates: str) -> str:
    """
    Fetches a 14-day agricultural weather forecast.
    You must pass the exact coordinates provided by the Geocoding Tool.

    Input Example:
        "44.837789,-0.57918" (A single string containing lat and lon separated by a comma)

    Output Example:
        "Day 1: Max 25.5°C, Min 12.0°C, Rain 0.0mm, Evapotranspiration 4.2mm
         Day 2: Max 28.1°C, Min 14.5°C, Rain 12.5mm, Evapotranspiration 3.1mm..."
    """
    try:
        lat, lon = coordinates.split(',')
        lat = lat.strip()
        lon = lon.strip()
    except ValueError:
        return "Error: Coordinates must be a comma-separated string like '44.84,-0.58'."

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "et0_fao_evapotranspiration"],
        "timezone": "auto",
        "forecast_days": 14
    }

    try:
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        data = response.json()

        daily = data.get('daily', {})
        dates = daily.get('time', [])
        max_temps = daily.get('temperature_2m_max', [])
        min_temps = daily.get('temperature_2m_min', [])
        rain = daily.get('precipitation_sum', [])
        et0 = daily.get('et0_fao_evapotranspiration', [])

        forecast_summary = f"14-Day Agricultural Forecast for Lat: {lat}, Lon: {lon}:\n"
        for i in range(len(dates)):
            forecast_summary += (
                f"Date {dates[i]}: "
                f"Max Temp {max_temps[i]}°C, "
                f"Min Temp {min_temps[i]}°C, "
                f"Rain {rain[i]}mm, "
                f"EvapoT {et0[i]}mm\n"
            )

        return forecast_summary

    except requests.exceptions.RequestException as e:
        return f"API Error occurred during weather fetch: {str(e)}"


if __name__ == "__main__":
    # Default transport is stdio.
    mcp.run()
