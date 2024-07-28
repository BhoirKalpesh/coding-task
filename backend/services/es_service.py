from elasticsearch import Elasticsearch
from config import ELASTICSEARCH_URL

es = Elasticsearch(ELASTICSEARCH_URL)

def query_candidates(job_description):
    results = es.search(index="candidates", body={"query": {"match": {"skills": job_description}}})
    return results['hits']['hits']
