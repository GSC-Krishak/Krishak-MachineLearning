from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import google.generativeai as genai
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress warnings
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)

# Load FastAPI app
app = FastAPI()

# Load models and index
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # Only used for query embeddings

with open("faiss_index.pkl", "rb") as f:
    index = pickle.load(f)

with open("structured_data.pkl", "rb") as f:
    structured_data = pickle.load(f)

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Initialize Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# Define request model
class CropRequest(BaseModel):
    query: str
    district: str
    state: str

# Function to retrieve top N matches
def retrieve_top_matches(query: str, top_n: int = 3):
    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)  # Ensure 2D shape
    D, I = index.search(query_embedding, top_n)
    return [structured_data[i] for i in I[0]]


# Function to determine the season
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

# Function to generate response with Gemini
def generate_response(query: str, district: str, state: str):
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
    Just return the name of the best crop (e.g., "Wheat") without explanations.
    """

    response = model.generate_content(prompt)
    return response.text

@app.post("/predict_crop")
async def predict_crop(request: CropRequest):
    try:
        response = generate_response(request.query, request.district, request.state)
        return {"recommended_crop": response.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
