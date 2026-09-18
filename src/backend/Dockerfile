FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image far smaller than the default CUDA wheels that
# sentence-transformers would otherwise pull in.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# backend/ holds the API + query path; rag.py is the shared embedding config that
# the ingest image builds the index with too.
COPY backend/ .
COPY rag.py .

# Bake the embedding model into the image so the backend does not download it
# on first query (and can run without network at startup).
ENV RAG_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ChromaDB is provided by the shared index volume (see CHROMA_DB_PATH).
ENV CHROMA_DB_PATH="/app/chroma_db"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
