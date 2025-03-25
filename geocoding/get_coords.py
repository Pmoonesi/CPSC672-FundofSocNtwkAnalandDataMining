# https://developers.google.com/maps/documentation/geocoding/requests-geocoding#component-filtering

import pandas as pd
import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()

API_KEY = os.getenv('GOOGLE_API_KEY')

df = pd.read_csv('../stations.csv')

base_url = "https://maps.googleapis.com/maps/api/geocode/json"

def save_response(data, name):
    with open(f"geolocation_data/{name}.json", "w") as f:
        f.write(
            json.dumps(data, indent=4, sort_keys=True)
        )

def get_lat_lon(id, address):
    response = requests.get(base_url, params={"address": address, "key": API_KEY})
    data = response.json()

    if data["status"] != "OK":
        save_response(data, f"ID#{id}")
        return None, None
    
    location = data["results"][0]["geometry"]["location"]

    if location is None:
        save_response(data, f"ID#{id}")
        return None, None
    
    return location["lat"], location["lng"]

def func(x):
    full_address = f"{x['Name'].strip()}, Toronto, ON"
    lat, lng = get_lat_lon(x['Id'], full_address)
    x['lat'] = lat
    x['lng'] = lng
    return x

df = df.apply(func, axis=1)
print(df)
df.to_csv('stations.csv')
