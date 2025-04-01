import json
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.encoders import jsonable_encoder
import subprocess
import os
from dotenv import load_dotenv  # Ensure dotenv is imported
from apscheduler.schedulers.background import BackgroundScheduler
import random

scheduler = BackgroundScheduler()

# Load environment variables
load_dotenv()


# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

#setting up fast api
app = FastAPI()

index = None
embed_model=None
api_key=None
model=None
chat_session=None
fertilizer_cost=None

def scheduled_task():
    #run setup.py
    subprocess.run(["python3","setup.py"])
    print("successfully ran setup.py")
    
#loading faiss on fastapi startup
@app.on_event("startup")
def load_faiss():
    global index,embed_model,api_key,model,chat_session,fertilizer_cost
    #faiss index
    index = faiss.read_index("crop_index.faiss")
    #embedding model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    # Configure Gemini API
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise ValueError("GENAI_API_KEY is not set in the environment variables.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
    chat_session = model.start_chat(history=[])
    # Fertilizer cost per unit
    fertilizer_cost = {"N": 20, "P": 25, "K": 15, "Mg": 30, "Calcium": 10}
    scheduler.add_job(scheduled_task, "cron", hour=10, minute=0)  # Run every day
    scheduler.start()
    print("started everything")

#unloading faiss on fastapi shutdown
@app.on_event("shutdown")
def unload_faiss():
    global index,embed_model,api_key,model,chat_session,fertilizer_cost
    index = None
    embed_model=None
    api_key=None
    model=None
    chat_session=None
    fertilizer_cost=None
    scheduler.shutdown()
    print('cleared everything')

# Define request model
class CropRecommendationRequest(BaseModel):
    n: float
    p: float
    k: float
    mg: float
    calcium: float
    ph: float
    previous_crops: list
    district: str
    state: str
    moisture: float
    soil_type: str


# Load structured data
with open("structured_data.json", "r") as f:
    structured_data = json.load(f)

# Function to retrieve top N matches
def retrieve_top_matches(query, top_n=5):
    query_embedding = embed_model.encode([query])
    D, I = index.search(np.array(query_embedding, dtype=np.float32), top_n)
    return [structured_data[i] for i in I[0]]

# Function to parse input query
def parse_query(query):
    values = {}
    matches = re.findall(r"(\w+):\s*([\d.]+)", query)
    for key, val in matches:
        values[key] = float(val)
    return values

# Function to calculate additional fertilizers required
def calculate_fertilizer_adjustment(crop, input_values):
    adjustments = {}
    total_fertilizer_cost = 0
    
    for key in ["N", "P", "K", "Mg", "Calcium"]:
        min_key = f"{key}_min"
        if min_key in crop:
            deficit = crop[min_key] - input_values.get(key, 0)
            if deficit > 0:
                adjustments[key] = deficit
                total_fertilizer_cost += deficit * fertilizer_cost[key]  # Calculate cost
    
    return adjustments, total_fertilizer_cost

# Function to check if nutrient values are within crop range
def is_within_range(crop, input_values):
    for key in ["N", "P", "K", "Mg", "Calcium"]:
        min_key, max_key = f"{key}_min", f"{key}_max"
        if min_key in crop and max_key in crop:
            if not (crop[min_key] <= input_values.get(key, 0) <= crop[max_key]):
                return False
    return True

# Function to determine the season based on the current date
def get_season():
    month = datetime.now().month
    return ["Winter", "Spring", "Summer", "Autumn"][(month // 3) % 4]

# Function to rank crops based on fertilizer requirement and profitability
def rank_crops(crops):
    crops.sort(key=lambda x: (x["Fertilizer_Cost"], -x["Profitability"]))
    return crops

# Function to generate response with Gemini
def extract_json(response_text):
    """Extracts and cleans JSON from Gemini's response."""
    match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)  # Extracts JSON inside backticks
    if match:
        json_str = match.group(1)
    else:
        json_str = response_text  # If no backticks, assume response is raw JSON

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return {"error": "AI response was not valid JSON"}

def generate_response(query, district, state, previous_crops, moisture=None, soil_type=None):
    season = get_season()
    input_values = parse_query(query)
    
    retrieved_data = retrieve_top_matches(query, top_n=10)

    ranked_crops = []
    for crop in retrieved_data:
        if crop["Commodity"] in previous_crops:  
            continue  

        if is_within_range(crop, input_values):
            fertilizer_adjustments, total_fertilizer_cost = calculate_fertilizer_adjustment(crop, input_values)
            profitability = crop["Modal Price"] - total_fertilizer_cost  

            crop["Fertilizer_Adjustments"] = fertilizer_adjustments
            crop["Fertilizer_Cost"] = total_fertilizer_cost
            crop["Profitability"] = profitability

            ranked_crops.append(crop)

    ranked_crops = rank_crops(ranked_crops)

    crop_suggestions = [
        {
            "Commodity": crop["Commodity"],
            "Profitability": crop["Profitability"],
            "Fertilizer_Cost": crop["Fertilizer_Cost"],
            "Fertilizer_Adjustments": crop["Fertilizer_Adjustments"]
        }
        for crop in ranked_crops
    ]

    prompt = f"""
    You are an expert in **Indian agriculture and crop recommendations**. A farmer has provided the following **soil and nutrient details**:  
    **"{query}"**  

    ### **Additional Context:**
    - **Location:** {district}, {state}  
    - **Current Season:** {season}  
    - **Moisture Content Percentage:** {moisture}
    - **Soil Type(to avoid crops with fertilizer requirements in range but not suitable to the location and soil):** {soil_type}
    - **Previously Grown Crops (Avoid for Rotation):** {', '.join(previous_crops) if previous_crops else ''}  
    - **Filtered Crops Based on Soil Data:** {crop_suggestions if crop_suggestions else "No direct match found"}  
    - **Latest Agricultural Insights:** Use up-to-date market prices, government policies, and farming trends to enhance recommendations.  

    ### **Your Task:**
    1️⃣ **Primary Goal:** Recommend the best crop that requires the least fertilizer adjustments and offers high profitability.  
    2️⃣ **If No Suitable Crop is Found:**  
      - Analyze market trends and soil compatibility.  
      - Suggest an **alternative high-yield, profitable crop** based on real-time insights.  
      - Mark this with **"(AI Recommended)"** if the suggestion is based on external data.  
    3️⃣ **Fertilizer Optimization:**  
      - List the fertilizer and profitability details for each recommended crop in numerical value.
      - List the fertilizers needed in **precise quantities** for optimal growth and 0 for the ones not required to add,but list all crops. 
      - If a nutrient is **deficient**, strictly suggest only the additional fertilizer's numerical value with precise quantities.  
      - If any of the nutrients is **excessive**, immediately stop processing further crop recommendation and only recommend "Grass" as the recommended crop.
    4️⃣ **Compatibility Ranking:**  
      - Each recommended crop should be ranked based on soil compatibility and profitability using the following scale:  
        - **Best** → Ideal compatibility and high profitability  
        - **Good** → Moderate compatibility and profitability  
        - **Not Best** → Requires adjustments but can still be grown  
        - **Not Recommended** → Poor compatibility or low profitability

    ### **Output Format (Strict JSON) – No Explanations,Strictly follow the format of the output i do not want any inconsistencies in my output format ,Only Valid JSON:**  
    Return a JSON object **without any extra text**.ranking at least **6 crops** from best to least suitable, Example format:
    ```json
    {{
        "Recommended_Crops": [
            {{
                "Commodity": "Wheat",
                "Profitability": 2500,
                "Fertilizer_Cost": 500,
                "Fertilizer_Adjustments": {{
                    "Nitrogen (N)": "Add 20 kg/ha",
                    "Phosphorus (P)": "Add 10 kg/ha",
                    "Magnesium (Mg)": "Add 5 ppm",
                    "Calcium (Ca)": "Add 5 ppm",
                    "Water content": "Add 10%"
                }},
                "Compatibility": "Good"
            }}
        ]
    }}
    ```
    """

    response = chat_session.send_message(prompt)

    return extract_json(response.text)  # Extract only the JSON

@app.post("/recommend")
def recommend_crop(request: CropRecommendationRequest):
    query = f"N: {request.n}, P: {request.p}, K: {request.k}, Mg: {request.mg}, Calcium: {request.calcium}, pH: {request.ph}"
    response = generate_response(query, request.district, request.state, request.previous_crops,request.moisture,request.soil_type)
    return JSONResponse(content=response)  # Always valid JSON

@app.get("/setup")
def setup_files():
    subprocess.run(["python3", "setup.py"]) 
    return {"message": "Setup completed successfully"}