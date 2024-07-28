from transformers import pipeline

generator = pipeline('text-generation', model='fine-tuned-model-path')

def generate_response(query):
    context = "context text based on the query"
    response = generator(f"Find top candidates for: {query}. Context: {context}", max_length=100)
    return response
