# index_data.py

from elasticsearch import Elasticsearch

es = Elasticsearch(
    [{'host': 'localhost', 'port': 9200, 'scheme': 'http'}]
)

# Create index
es.indices.create(index='candidates', ignore=400)

# Sample data to index
sample_data = [
    {
        "name": "John Doe",
        "skills": ["HTML", "CSS", "JavaScript", "React"],
        "experience": "5 years",
        "profile": "Experienced UI Developer."
    },
    {
        "name": "Jane Smith",
        "skills": ["HTML", "CSS", "JavaScript", "Angular"],
        "experience": "4 years",
        "profile": "Front-end developer with experience in Angular."
    }
]

# Index data
for i, doc in enumerate(sample_data):
    es.index(index='candidates', id=i, document=doc)
