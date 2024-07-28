# rag_framework.py

from elasticsearch import Elasticsearch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize Elasticsearch with complete configuration
es = Elasticsearch(
    [{'host': 'localhost', 'port': 9200, 'scheme': 'http'}]
)

model = GPT2LMHeadModel.from_pretrained('path/to/fine-tuned-model')
tokenizer = GPT2Tokenizer.from_pretrained('path/to/fine-tuned-model')

def search_candidates(query):
    # Search in Elasticsearch
    res = es.search(index="candidates", body={
        "query": {
            "match": {
                "profile": query
            }
        }
    })
    return res['hits']['hits']

def generate_response(candidates):
    # Generate response with LLM
    inputs = tokenizer.encode("Top candidates are: " + str(candidates), return_tensors='pt')
    outputs = model.generate(inputs, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def process_query(job_description):
    candidates = search_candidates(job_description)
    response = generate_response(candidates)
    return response
