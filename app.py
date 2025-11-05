from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

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
index = faiss.IndexFlatL2(dimension)
index.add(np.array(question_embeddings))

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    user_embedding = model.encode([user_message])

    D, I = index.search(np.array(user_embedding), k=1)
    best_answer = answers[I[0][0]]

    return jsonify({"reply": best_answer})

if __name__ == "__main__":
    app.run(debug=True)