import json
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import logging
import subprocess

subprocess.run(["python3", "scraper.py"])  

# Suppress unnecessary logs
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)

# Load pre-trained embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

generation_config = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Load JSON data
with open("standardized_crop_data.json", "r") as file:
    crop_data = json.load(file)

# Fertilizer cost per unit
fertilizer_cost = {"N": 20, "P": 25, "K": 15, "Mg": 30, "Calcium": 10}

# Extract structured crop details
def extract_crop_details(crop):
    return {
        "Commodity": crop["Commodity"],
        "Modal Price": crop["Modal Price(Price in Rs.)"], #float(crop["Modal Price(Price in Rs.)"].replace(",", "")),  # Convert string price to float
        "N_min": crop["N (kg/ha)_min"], "N_max": crop["N (kg/ha)_max"],
        "P_min": crop["P (kg/ha)_min"], "P_max": crop["P (kg/ha)_max"],
        "K_min": crop["K (kg/ha)_min"], "K_max": crop["K (kg/ha)_max"],
        "Mg_min": crop["Mg (ppm)_min"], "Mg_max": crop["Mg (ppm)_max"],
        "Calcium_min": crop["Calcium (ppm)_min"], "Calcium_max": crop["Calcium (ppm)_max"],
    }

structured_data = [extract_crop_details(crop) for crop in crop_data]

# Generate embeddings for FAISS
text_descriptions = [
    f"{crop['Commodity']} N: {crop['N_min']}-{crop['N_max']}, P: {crop['P_min']}-{crop['P_max']}, K: {crop['K_min']}-{crop['K_max']}, Mg: {crop['Mg_min']}-{crop['Mg_max']}, Calcium: {crop['Calcium_min']}-{crop['Calcium_max']}"
    for crop in structured_data
]

embeddings = embed_model.encode(text_descriptions)

# Convert embeddings to FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings, dtype=np.float32))

# Save FAISS index
faiss.write_index(index, "crop_index.faiss")

# Save structured data
with open("structured_data.json", "w") as f:
    json.dump(structured_data, f)
