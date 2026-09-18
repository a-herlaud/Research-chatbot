COMPOSE = docker compose -f src/docker-compose.yml

.PHONY: up build down logs restart rag-ingest rag-ingest-docker

up:
	$(COMPOSE) up --build -d

build:
	$(COMPOSE) build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

restart: down up

# Build the RAG index from research_papers/ using uv on the host.
rag-ingest:
	@mkdir -p chroma_db
	uv run ingest_papers.py

# Same thing, but inside the backend image (model already baked in).
rag-ingest-docker:
	@mkdir -p chroma_db
	$(COMPOSE) --profile tools run --rm ingest
