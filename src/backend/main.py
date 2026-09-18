import os

import litellm
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import rag

load_dotenv()

app = FastAPI(title="Research Chatbot Backend")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
if "/" not in GEMINI_MODEL:
    GEMINI_MODEL = f"gemini/{GEMINI_MODEL}"


class ChatRequest(BaseModel):
    message: str


class Source(BaseModel):
    paper_id: str
    title: str
    score: float


class ChatResponse(BaseModel):
    reply: str
    sources: list[Source] = []


SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using ONLY the "
    "numbered excerpts provided below. When you use an excerpt, cite it inline "
    "as [paper_id]. If the excerpts do not contain the answer, say so plainly "
    "instead of inventing information."
)


def build_augmented_prompt(question: str, chunks: list[dict]) -> str:
    context_blocks = [
        f"[{index}] (paper_id={chunk['paper_id']}, title={chunk['title']})\n{chunk['text']}"
        for index, chunk in enumerate(chunks, start=1)
    ]
    context = "\n\n".join(context_blocks)
    return (
        f"Excerpts from retrieved research papers:\n\n{context}\n\n"
        f"Question: {question}"
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": rag.get_collection(create=False) is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    chunks = rag.retrieve(request.message, rag.TOP_K)
    if chunks:
        user_content = build_augmented_prompt(request.message, chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    else:
        # No index yet (or empty): answer without grounding rather than failing.
        messages = [{"role": "user", "content": request.message}]

    try:
        response = litellm.completion(
            model=GEMINI_MODEL,
            messages=messages,
            api_key=os.getenv("GEMINI_API_KEY"),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model request failed: {exc}")

    sources = [
        Source(paper_id=c["paper_id"], title=c["title"], score=c["score"])
        for c in chunks
    ]
    return ChatResponse(reply=response.choices[0].message.content, sources=sources)
