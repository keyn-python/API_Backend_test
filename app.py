from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return 'Server is running pog!'

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    message = data.get("message", "").strip().lower()
    if message == "hi":
        return jsonify({"response": "hello"})
    else:
        return jsonify({"response": "I didn't understand that."})

if __name__ == "__main__":
    app.run(debug=True)
