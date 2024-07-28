from flask import Blueprint, request, jsonify
from services.es_service import query_candidates
from services.llm_service import generate_response

query_bp = Blueprint('query_bp', __name__)

@query_bp.route('/query', methods=['POST'])
def handle_query():
    job_description = request.json.get('job_description')
    candidates = query_candidates(job_description)
    response = generate_response(job_description)
    return jsonify({'candidates': candidates, 'response': response})
