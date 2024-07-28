from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_generative_model(model_name='facebook/bart-large'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./models_cache')
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='./models_cache')
    return tokenizer, model


def generate_summary(tokenizer, model, job_description, candidates):
    context = f"Job Description: {job_description}\nCandidates:\n"
    for candidate in candidates:
        context += f"- {candidate['Name']}: {candidate['skills_processed']}\n"
    
    inputs = tokenizer.encode(context, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary
