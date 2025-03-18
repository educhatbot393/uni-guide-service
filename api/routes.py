# api/routes.py
from flask import Blueprint, request, jsonify
from services.document_service import process_documents
from services.embedding_service import initialize_vectorstore, get_vectorstore
from services.query_service import process_query
import time

api_bp = Blueprint('api', __name__)

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@api_bp.route('/initialize', methods=['POST'])
def initialize_data():
    """Initialize vector store with document data"""
    try:
        # Initialize Pinecone and process documents
        if initialize_vectorstore():
            return jsonify({
                "status": "success",
                "message": "Vector store initialized successfully"
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to initialize vector store"
            }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@api_bp.route('/ask', methods=['POST'])
def ask_question():
    """Process user query and return response"""
    data = request.json
    if not data or 'query' not in data:
        return jsonify({
            "status": "error",
            "message": "Query parameter is required"
        }), 400
        
    query = data['query']
    conversation_history = data.get('conversation_history', [])
    
    # Check if vector store is initialized
    if not get_vectorstore():
        return jsonify({
            "status": "error",
            "message": "Vector store not initialized. Call /api/initialize first."
        }), 400
    
    try:
        # Process query and measure response time
        start_time = time.time()
        response = process_query(query, conversation_history)
        elapsed_time = time.time() - start_time
        
        return jsonify({
            "status": "success",
            "response": response,
            "response_time": f"{elapsed_time:.2f}"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500