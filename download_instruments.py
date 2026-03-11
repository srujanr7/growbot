from growwapi import GrowwAPI
import pandas as pd

# load access token
with open("access_token.txt") as f:
    token = f.read().strip()

api = GrowwAPI(token)

print("Downloading instruments from Groww...")

df = api.get_all_instruments()

df.to_csv("instruments.csv", index=False)

print(f"Saved instruments.csv with {len(df)} rows")
