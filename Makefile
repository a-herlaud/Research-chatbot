COMPOSE = docker compose -f src/docker-compose.yml

.PHONY: up build down logs restart rag-ingest

up:
	$(COMPOSE) up --build -d

build:
	$(COMPOSE) build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

restart: down up

# Create the RAG index in the dedicated ingest image, writing to the shared
# ./chroma_db volume that the backend reads.
rag-ingest:
	@mkdir -p chroma_db
	$(COMPOSE) --profile tools run --rm ingest
