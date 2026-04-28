# test_geocoding.py
import requests

def test_nominatim(location_name):
    print(f"Testing Geocoding for: {location_name}")
    url = "https://nominatim.openstreetmap.org/search"
    
    # Nominatim requires a User-Agent to track usage and prevent abuse
    headers = {
        "User-Agent": "test-bot"
    }
    params = {
        "q": location_name,
        "format": "json",
        "limit": 1
    }

    response = requests.get(url, headers=headers, params=params, verify=False)
    
    if response.status_code == 200 and len(response.json()) > 0:
        data = response.json()[0]
        lat = data.get('lat')
        lon = data.get('lon')
        print(f"Success! Coordinates: Latitude {lat}, Longitude {lon}")
        return lat, lon
    else:
        print(f"Failed. Status Code: {response.status_code}")
        print("Response:", response.text)
        return None, None

if __name__ == "__main__":
    test_nominatim("Patna, India")