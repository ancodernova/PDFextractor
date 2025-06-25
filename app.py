from flask import Flask, request, jsonify
import os
from utils import load_and_vectorize, create_qa_chain, save_to_db, load_faiss_if_exists
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initial Load
vector_db = load_faiss_if_exists()
qa_chain = create_qa_chain(vector_db) if vector_db else None

@app.route("/upload_pdf", methods=["POST"])
def upload_pdf():
    global vector_db, qa_chain

    if "pdf" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["pdf"]
    pdf_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(pdf_path)

    vector_db = load_and_vectorize(pdf_path)
    qa_chain = create_qa_chain(vector_db)

    return jsonify({"message": "PDF uploaded and vectorized successfully!"})

@app.route("/ask", methods=["POST"])
def ask_question():
    global qa_chain
    if qa_chain is None:
        return jsonify({"error": "No PDF uploaded yet"}), 400

    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    result = qa_chain(question)
    answer = result["result"]
    sources = result.get("source_documents", [])

    source_info = []
    for doc in sources:
        page = doc.metadata.get("page", "N/A")
        snippet = doc.page_content.strip().replace("\n", " ")[:200]
        source_info.append(f"Page {page}: {snippet}...")

    return jsonify({
        "question": question,
        "answer": answer,
        "sources": source_info
    })

@app.route("/save_answer", methods=["POST"])
def save_answer():
    data = request.json
    question = data.get("question")
    answer = data.get("answer")
    sources = data.get("sources")

    if not all([question, answer, sources]):
        return jsonify({"error": "Missing fields"}), 400

    save_to_db(question, answer, "\n".join(sources))
    return jsonify({"message": "Answer saved successfully!"})

if __name__ == "__main__":
    app.run(debug=True)
