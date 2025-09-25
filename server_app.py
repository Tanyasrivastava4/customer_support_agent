from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import json

app = FastAPI(title="Salad AI Server with LLM")

# -----------------------
# Load KB from JSON
# -----------------------
with open("kb.json", "r") as f:
    KB = json.load(f)

# -----------------------
# Pydantic Model
# -----------------------
class TextInput(BaseModel):
    text: str

# -----------------------
# Load Hugging Face LLM for text generation
# -----------------------
MODEL_NAME = "google/flan-t5-small"  # Lightweight CPU-friendly
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# -----------------------
# Email Classification using LLM
# -----------------------
@app.post("/classify")
def classify_email(data: TextInput):
    prompt = f"Classify this email into one of the categories: billing, technical, account, refund, shipping, general.\nEmail: {data.text}"
    result = generator(prompt, max_length=20)[0]['generated_text']
    category = result.strip().lower()
    if category not in KB.keys():
        category = "general"
    return {"category": category}

# -----------------------
# RAG: Retrieve Answer from KB using LLM
# -----------------------
@app.post("/rag")
def get_rag_answer(data: TextInput):
    # First classify email
    category = classify_email(data).get("category")
    # Generate better response using LLM + KB
    kb_answer = KB.get(category, "Sorry, we do not have an answer for this query yet.")
    prompt = f"Generate a professional customer support response based on the following answer:\n{kb_answer}\nEmail: {data.text}"
    response = generator(prompt, max_length=200)[0]['generated_text']
    return {"answer": response}
