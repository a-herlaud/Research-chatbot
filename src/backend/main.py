import os

import litellm
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Research Chatbot Backend")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
if "/" not in GEMINI_MODEL:
    GEMINI_MODEL = f"gemini/{GEMINI_MODEL}"


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        response = litellm.completion(
            model=GEMINI_MODEL,
            messages=[{"role": "user", "content": request.message}],
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model request failed: {exc}")
    return ChatResponse(reply=response.choices[0].message.content)
