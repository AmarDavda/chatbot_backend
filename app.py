import json
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
from flask_cors import CORS
import time

app = Flask(__name__)
# Enable CORS for all domains (mobile apps)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables
model = None
index = None
questions = []
answers = []

def load_resources():
    """Load AI resources"""
    global model, index, questions, answers
    
    try:
        print("🔄 Loading AI resources...")
        
        # Load dataset
        with open("faq_dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = [item["question"] for item in data]
        answers = [item["answer"] for item in data]

        # Load model - using smaller model for faster mobile response
        model = SentenceTransformer('all-MiniLM-L6-v2')
        question_embeddings = model.encode(questions)

        # Create FAISS index
        dimension = question_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(question_embeddings))
        
        print("✅ AI resources loaded successfully!")
        print(f"📊 Loaded {len(questions)} Q&A pairs")
        
    except Exception as e:
        print(f"❌ Error loading resources: {str(e)}")
        raise e

@app.route("/")
def home():
    return jsonify({
        "message": "🤖 Chatbot API is running!",
        "status": "active",
        "version": "1.0",
        "mobile_support": True
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "total_questions": len(questions),
        "timestamp": time.time()
    })

@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    # Handle CORS preflight
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    
    try:
        data = request.get_json()
        
        if not data or "message" not in data:
            return _cors_response(jsonify({
                "error": "Message is required",
                "code": "MISSING_MESSAGE"
            }), 400)
        
        user_message = data["message"].strip()
        if not user_message:
            return _cors_response(jsonify({
                "error": "Message cannot be empty",
                "code": "EMPTY_MESSAGE"
            }), 400)
        
        # Process the message
        user_embedding = model.encode([user_message])
        D, I = index.search(np.array(user_embedding), k=1)
        
        if I[0][0] >= len(answers):
            response_data = {
                "reply": "I'm sorry, I couldn't find an answer to that question. Please try rephrasing.",
                "confidence": 0.0,
                "status": "not_found"
            }
        else:
            confidence = float(1 - D[0][0]) if D[0][0] > 0 else 1.0
            response_data = {
                "reply": answers[I[0][0]],
                "original_question": user_message,
                "confidence": confidence,
                "status": "success"
            }
        
        return _cors_response(jsonify(response_data))
        
    except Exception as e:
        return _cors_response(jsonify({
            "error": "Internal server error",
            "code": "SERVER_ERROR",
            "details": str(e)
        }), 500)

def _build_cors_preflight_response():
    response = jsonify({"status": "preflight"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    return response

def _cors_response(response, status_code=200):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response, status_code

# Load resources when app starts
@app.before_first_request
def initialize():
    load_resources()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
