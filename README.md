# Crop Recommendation System

## Overview
The Crop Recommendation System is an AI-powered tool that helps farmers select the most economically viable and nutritionally suitable crops based on real-time market prices, soil nutrient levels, and seasonal conditions. It leverages advanced machine learning, NLP models, and FAISS-based similarity search to provide intelligent crop recommendations.

## Key Features
- **Real-time Market Data Integration**: Fetches crop prices from the eNAM portal to ensure economically beneficial recommendations.
- **Soil Nutrient Analysis**: Compares user-input soil nutrient levels with ideal crop requirements.
- **AI-Powered Crop Matching**: Utilizes FAISS and sentence-transformers to find the best crop matches based on soil data.
- **Fertilizer Optimization**: Suggests necessary fertilizer adjustments to achieve optimal soil conditions for selected crops.
- **Intelligent Crop Rotation**: Avoids suggesting crops recently grown in the specified location to promote soil health.
- **Seasonal Suitability**: Recommends crops based on the current season to maximize yield and profitability.
- **Conversational AI Support**: Uses Google Gemini AI to generate structured recommendations and insights.
- **FAST API Integration**: Provides an endpoint to receive soil parameters and return optimized crop suggestions in JSON format.

## How It Works
1. Fetches live crop price data from the eNAM portal.
2. Processes and standardizes crop nutrient requirements.
3. Embeds crop information using sentence-transformers for similarity search.
4. Stores embeddings in a FAISS index for efficient retrieval.
5. Receives user-input soil parameters via a FastAPI endpoint.
6. Retrieves the most suitable crops based on nutrient compatibility and economic viability.
7. Adjusts fertilizer recommendations and calculates costs.
8. Uses Gemini AI to refine final suggestions and provide structured JSON responses.

## Use Cases
- Farmers looking for data-driven crop selection.
- Agricultural researchers analyzing soil suitability for various crops.
- Government agencies optimizing crop recommendations for different regions.
- Agri-tech platforms integrating AI-powered recommendations for farmers.

## Technologies Used
- **Python** (FastAPI, Pandas, NumPy, Requests)
- **Machine Learning** (FAISS, Sentence-Transformers)
- **Google Gemini AI** for conversational insights
- **Real-time Data Processing** (eNAM API integration)
- **JSON-based REST API** for seamless integration

