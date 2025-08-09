"""
Next.js interaction example (i think?)
fetch('http://localhost:8000/api/match-data')
  .then(res => res.json())
  .then(data => console.log(data))

  ^ should work
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import uvicorn

app = FastAPI()

# Enable CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/match-data")
def send_json_file():
    """
    Read JSON file from disk and send to Next.js
    """
    try:
        # Read the JSON file
        with open("match_data.json", "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="match_data.json not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Invalid JSON file")

if __name__ == "__main__":
    print("API running at http://localhost:8000")
    print("Next.js can fetch from: http://localhost:8000/api/match-data")
    uvicorn.run(app, host="0.0.0.0", port=8000)