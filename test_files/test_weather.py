# test_weather.py
import requests

def test_open_meteo(lat, lon):
    print(f"Testing Open-Meteo for Lat: {lat}, Lon: {lon}")
    url = "https://api.open-meteo.com/v1/forecast"
    
    # Fetching 14-day daily data: Max Temp, Min Temp, and Precipitation
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
        "timezone": "auto",
        "forecast_days": 14
    }

    response = requests.get(url, params=params, verify=False)
    
    if response.status_code == 200:
        data = response.json()
        print(data)
        print("Success! Here is a sample of the returned data:")
        
        # Print the first 3 days of forecast
        daily_dates = data['daily']['time'][:3]
        max_temps = data['daily']['temperature_2m_max'][:3]
        
        for date, temp in zip(daily_dates, max_temps):
            print(f"Date: {date} | Max Temp: {temp}°C")
    else:
        print(f"Failed. Status Code: {response.status_code}")
        print("Response:", response.text)

if __name__ == "__main__":
    test_open_meteo("25.6093239", "85.1235252")