import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle

# Load candidate dataset
candidate_data = pd.read_csv('candidates.csv')

# Preprocess text (if needed)
def preprocess(text):
    return text.lower()

candidate_data['skills_processed'] = candidate_data['Job Skills'].apply(preprocess)

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode candidate skills to get embeddings
candidate_embeddings = model.encode(candidate_data['skills_processed'].tolist(), convert_to_tensor=True)

# Save embeddings and candidate data
with open('candidate_embeddings.pkl', 'wb') as f:
    pickle.dump((candidate_data, candidate_embeddings), f)

print("Model training and embedding saving completed.")
