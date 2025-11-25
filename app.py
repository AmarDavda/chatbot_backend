import json
import numpy as np
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss

app = Flask(__name__)

# Load dataset
with open("faq_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# Load model and encode questions
model = SentenceTransformer('all-mpnet-base-v2')
question_embeddings = model.encode(questions)

# Create FAISS index
dimension = question_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # Fixed typo here
index.add(np.array(question_embeddings))

@app.route("/")
def home():
    return """
    <html>
        <head><title>Chatbot API</title></head>
        <body>
            <h1>🤖 Chatbot is Running!</h1>
            <p>Use POST /chat with JSON {"message": "your question"}</p>
        </body>
    </html>
    """

@app.route("/chat", methods=["POST"])  # Fixed methods parameter
def chat():
    user_message = request.json["message"]
    user_embedding = model.encode([user_message])
    D, I = index.search(np.array(user_embedding), k=1)
    best_answer = answers[I[0][0]]  # Fixed indexing

    return jsonify({"reply": best_answer})

@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": True})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)  # Added host and port for Render
