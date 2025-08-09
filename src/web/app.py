from flask import Flask, render_template, jsonify
import os
import json
from src import config

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def data():
    output_file = os.path.join(config.OUTPUT_DIR, "tracked_data.json")
    with open(output_file, "r") as f:
        data = json.load(f)
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
