from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime
import google.generativeai as genai
import os
import logging
import torch
import random

# Set seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Suppress warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)

# Load saved models and index
device = "mps" if torch.backends.mps.is_available() else "cpu"
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

with open("faiss_index.pkl", "rb") as f:
    index = pickle.load(f)

with open("structured_data.pkl", "rb") as f:
    structured_data = pickle.load(f)

# Configure Gemini API
genai.configure(api_key="AIzaSyDEthHdx0TfOC0A_wIQsAWoQTXyPo-EcsM")  # Replace with actual API key
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

app = FastAPI()

# Define request model
class CropRecommendationRequest(BaseModel):
    query: str
    district: str
    state: str

def retrieve_top_matches(query, top_n=3):
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding, dtype=np.float32), top_n)
    return [structured_data[i] for i in I[0]]

def get_season():
    month = datetime.now().month
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

def generate_response(query, district, state):
    season = get_season()
    retrieved_data = retrieve_top_matches(query, top_n=3)
    prompt = f"""
    You are an expert in Indian agriculture and crop recommendations. A user has provided the following **soil and nutrient details:** 
    "{query}"
    
    **Additional Context:**
    - **Location:** {district}, {state}
    - **Current Season:** {season}
    - **User's Knowledge Base (Retrieved Crops & Market Trends):** {retrieved_data}
    - **Your Own Up-to-Date Knowledge (as of today):** Use the latest agricultural insights, government policies, and real-time market trends.
    
    **Your Task:**
    - Analyze the **best crops** based on soil data, climate, and market conditions.
    - Use **both the provided knowledge base** and **your latest knowledge** to ensure accuracy.
    - Prioritize crops that are **profitable, sustainable, and well-suited for this region**.
    - If multiple crops are viable, compare them based on **profitability, ease of farming, and sustainability**.
    - Provide an **actionable recommendation** with clear reasoning.
    
    **Output Format:**  
    Just return the name of the 3 best crops (e.g., "Wheat") without explanations.
    """
    response = model.generate_content(prompt)
    return response.text

@app.post("/recommend")
def recommend_crop(request: CropRecommendationRequest):
    response = generate_response(request.query, request.district, request.state)
    return {"recommended_crops": response.split("\n")}
