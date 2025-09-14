from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from src.inference import PunctuationRestorer

# Initialize FastAPI app
app = FastAPI(title="Punctuation Restoration API")

# Allow CORS (so frontend JS can call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to ["http://localhost:8000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
restorer = PunctuationRestorer(model_path="checkpoints/best_model.pt")

# Request schema
class TextInput(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Punctuation Restoration API is running!"}

@app.post("/restore")
def restore_text(input_data: TextInput):
    restored = restorer.restore_punctuation(input_data.text)
    return {"input": input_data.text, "restored": restored}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
