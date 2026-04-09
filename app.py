from flask import Flask, request, jsonify, render_template
from rag import get_answer

app = Flask(__name__)

queries = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    query = request.json["query"]
    queries.append(query)

    answer = get_answer(query)
    return jsonify({"answer": answer})

@app.route("/stats")
def stats():
    return {"total_queries": len(queries)}

if __name__ == "__main__":
    app.run(debug=True)
