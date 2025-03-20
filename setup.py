import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load pre-trained embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load JSON data
with open("updated_structured_tables.json", "r") as file:
    crop_data = json.load(file)

# Extract structured crop details
def extract_crop_details(crop):
    crop_name = crop["heading"]
    page = crop["page"]
    details = []
    
    for table in crop["tables"]:
        for row in table["data"]:
            formatted_text = f"Crop: {crop_name}, Page: {page}, "
            formatted_text += ", ".join([f"{key}: {value}" for key, value in row.items() if value])
            details.append(formatted_text)
    
    return details

structured_data = []
for crop in crop_data:
    structured_data.extend(extract_crop_details(crop))

# Generate embeddings
embeddings = embed_model.encode(structured_data)

# Convert embeddings to FAISS index
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings, dtype=np.float32))

# Save FAISS index and structured data
with open("faiss_index.pkl", "wb") as f:
    pickle.dump(index, f)

with open("structured_data.pkl", "wb") as f:
    pickle.dump(structured_data, f)

print("✅ Setup complete! FAISS index and structured data saved.")