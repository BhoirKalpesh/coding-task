from flask import Flask, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from flask_cors import CORS
from utils.faiss_indexing import build_faiss_index, load_faiss_index
from models.rag_model import load_generative_model, generate_summary
import numpy as np

app = Flask(__name__)
CORS(app)

# Load candidate dataset
candidate_data3 = pd.read_csv('../data/candidates.csv')

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocess text (if needed)
def preprocess(text):
    return text.lower()

candidate_data3['skills_processed'] = candidate_data3['Job Skills'].apply(preprocess)

# Encode candidate skills
candidate_embeddings = model.encode(candidate_data3['skills_processed'].tolist(), convert_to_tensor=False)

# Create a FAISS index
index = build_faiss_index(candidate_embeddings)

# Load generative model
tokenizer, generative_model = load_generative_model()

# Function to find top matching candidates
def find_top_candidates(job_description, candidate_data, top_n=10):
    job_description_processed = preprocess(job_description)
    job_description_embedding = model.encode(job_description_processed, convert_to_tensor=False)
    D, I = index.search(np.array([job_description_embedding]), top_n)
    top_candidates = candidate_data.iloc[I[0]]
    return top_candidates

@app.route('/match', methods=['POST'])
def query():
    data = request.json
    job_description = data.get('job_description', '')
    if not job_description:
        return jsonify({"error": "Please enter a job description"}), 400

    top_candidates = find_top_candidates(job_description, candidate_data3)
    candidates_json = top_candidates.to_dict(orient='records')
    summary = generate_summary(tokenizer, generative_model, job_description, candidates_json)
    return jsonify({"response": candidates_json, "summary": summary})

if __name__ == '__main__':
    app.run(debug=True)
