import requests
import pandas as pd
from datetime import datetime
import numpy as np
import json

url = "https://enam.gov.in/web/Liveprice_ctrl/trade_data_list"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Content-Type": "application/x-www-form-urlencoded"
}

payload = {
    "language": "en",
    "stateName": "-- All --",
    "fromDate": datetime.today().strftime('%Y-%m-%d'),
    "toDate": datetime.today().strftime('%Y-%m-%d'),
}

response = requests.post(url, headers=headers, data=payload)

if response.status_code == 200:
    try:
        data = response.json()
        if isinstance(data, dict) and "data" in data:
            records = data["data"]

            # Normalize JSON data into a DataFrame
            df = pd.json_normalize(records)

            # Select and rename required columns
            df_filtered = df[["apmc", "commodity", "min_price", "modal_price", "max_price"]]
            df_filtered.rename(columns={
                "apmc": "APMC's",
                "commodity": "Commodity",
                "min_price": "Min Price(Price in Rs.)",
                "modal_price": "Modal Price(Price in Rs.)",
                "max_price": "Max Price(Price in Rs.)"
            }, inplace=True)

            # Save to CSV
            filename = "enam_live_prices.csv"
            df_filtered.to_csv(filename, index=False)

            print(f"Data saved as {filename}")
        else:
            print("No valid data found.")
    except Exception as e:
        print(f" Error parsing JSON: {e}")
else:
    print(f" Failed to fetch data. Status Code: {response.status_code}")

df1=pd.read_csv('enam_live_prices.csv')

df1 = df1.drop_duplicates(subset=['Commodity'])

# df1.to_csv('enam_live_prices_updated.csv')

df2=pd.read_csv('crop_npk_data.csv')

df_merged = pd.merge(df1, df2, on="Commodity")

df_merged.drop(["APMC's"],axis=1,inplace=True)

columns=['Commodity',
       'Min Price(Price in Rs.)', 'Modal Price(Price in Rs.)',
       'Max Price(Price in Rs.)', 'N (kg/ha)', 'P (kg/ha)', 'K (kg/ha)', 'Mg (ppm)', 'Calcium (ppm)',
       'Water content (%)']

df_merged.columns=columns

# Fill NaN values with 1
df_merged.fillna(1, inplace=True)

# Function to process nutrient values and ensure a consistent format
def process_value(value):
    if isinstance(value, str) and "-" in value:  # If it's a range
        parts = value.split("-")
        return {"min": float(parts[0]), "max": float(parts[1])}
    elif isinstance(value, (int, float)) and not np.isnan(value):  # If it's a single number
        return {"min": value, "max": value}
    elif isinstance(value, str) and value.isnumeric():  # If it's a string number
        num = float(value)
        return {"min": num, "max": num}
    else:  # If value is NaN or unrecognized
        return {"min": 1, "max": 1}

# Columns to convert
nutrient_columns = [
    "N (kg/ha)", "P (kg/ha)", "K (kg/ha)", "Mg (ppm)", "Calcium (ppm)", "Water content (%)"
]

# Convert nutrient columns
for col in nutrient_columns:
    df_merged[col] = df_merged[col].apply(lambda x: process_value(x))

# Convert to JSON format
json_data = []
for _, row in df_merged.iterrows():
    crop_entry = {
        "Commodity": row["Commodity"], 
        "Min Price(Price in Rs.)": row["Min Price(Price in Rs.)"],
        "Modal Price(Price in Rs.)": row["Modal Price(Price in Rs.)"], 
        "Max Price(Price in Rs.)": row["Max Price(Price in Rs.)"]
    }
    
    # Add nutrients in a standardized format
    for col in nutrient_columns:
        crop_entry[f"{col}_min"] = row[col]["min"]
        crop_entry[f"{col}_max"] = row[col]["max"]
    
    json_data.append(crop_entry)

# Save as JSON
with open("standardized_crop_data.json", "w") as f:
    json.dump(json_data, f, indent=4)

print("✅ Standardized JSON saved successfully! (NaNs replaced with 1)")
