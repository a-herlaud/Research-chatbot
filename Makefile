COMPOSE = docker compose -f src/docker-compose.yml

.PHONY: up build down logs restart

up:
	$(COMPOSE) up --build -d

build:
	$(COMPOSE) build

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

restart: down up
