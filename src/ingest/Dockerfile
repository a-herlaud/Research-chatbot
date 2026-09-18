# Dedicated, self-contained image that creates the RAG index. Built with src/ as
# the context (see src/docker-compose.yml) and copies only files under src/.
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image far smaller than the default CUDA wheels.
COPY ingest/requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# rag.py is the shared embedding/chroma config also used by the backend.
COPY rag.py .
COPY ingest/ingest_papers.py ./ingest_papers.py

# Bake the embedding model into the image so indexing needs no network at run.
ENV RAG_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Paths inside the container; docker-compose mounts the shared volume/dirs here.
ENV CHROMA_DB_PATH="/data/chroma"
CMD ["python", "ingest_papers.py", "--papers-dir", "/data/research_papers"]
