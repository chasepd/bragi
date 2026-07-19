.PHONY: help start stop restart clean compose-build compose-up compose-down compose-logs compose-ps compose-restart

.DEFAULT_GOAL := help

BRAGI := uv run bragi
COMPOSE_PROJECT_NAME := bragi-prod
COMPOSE := docker compose --project-name $(COMPOSE_PROJECT_NAME)
COMPOSE_SERVICE := bragi
STATIC_DIR := bragi_web/static

help:
	@printf '%s\n' \
		"Available targets:" \
		"  help             Show available targets." \
		"  start            Build the frontend and start the local static web app." \
		"  stop             Stop the local web app." \
		"  restart          Stop, clean static assets, and start again." \
		"  clean            Remove built static frontend assets." \
		"  compose-build    Build the production Compose service image." \
		"  compose-up       Build and start the production Compose service." \
		"  compose-down     Stop and remove the production Compose service." \
		"  compose-logs     Follow production Compose service logs." \
		"  compose-ps       Show production Compose service status." \
		"  compose-restart  Restart the production Compose service."

start:
	$(BRAGI) start --frontend-mode static --build-frontend

stop:
	$(BRAGI) stop

restart: stop clean start

clean:
	rm -rf $(STATIC_DIR)

compose-build:
	$(COMPOSE) build $(COMPOSE_SERVICE)

compose-up:
	$(COMPOSE) up --build -d $(COMPOSE_SERVICE)

compose-down:
	$(COMPOSE) down

compose-logs:
	$(COMPOSE) logs -f $(COMPOSE_SERVICE)

compose-ps:
	$(COMPOSE) ps

compose-restart: compose-down compose-up
