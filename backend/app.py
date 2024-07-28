from flask import Flask, request, jsonify
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, util
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Load candidate dataset
candidate_data3 = pd.read_csv(r'D:\ML\assignments\video\data\candidates.csv')

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Preprocess text (if needed)
def preprocess(text):
    return text.lower()

candidate_data3['skills_processed'] = candidate_data3['Job Skills'].apply(preprocess)

# Function to find top matching candidates
def find_top_candidates(job_description, candidate_data, top_n=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_description_processed = preprocess(job_description)
    job_description_embedding = model.encode(job_description_processed, convert_to_tensor=True)
    candidate_embeddings = model.encode(candidate_data['skills_processed'].tolist(), convert_to_tensor=True)

    similarity_scores = util.pytorch_cos_sim(job_description_embedding, candidate_embeddings)
    top_indices = similarity_scores.argsort(dim=1, descending=True)[0][:top_n]
    top_candidates = candidate_data.iloc[top_indices]

    return top_candidates

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    job_description = data.get('job_description', '')
    top_candidates = find_top_candidates(job_description, candidate_data3)
    candidates_json = top_candidates.to_dict(orient='records')
    return jsonify({"response": candidates_json})

if __name__ == '__main__':
    app.run(debug=True)
