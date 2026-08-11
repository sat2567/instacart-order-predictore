.PHONY: up down logs pipeline test lint demo clean-db artifacts

up:            ## build and start db -> pipeline -> api
	docker compose up --build -d
	docker compose logs -f pipeline

down:
	docker compose down

logs:
	docker compose logs -f api

pipeline:      ## re-run the pipeline (FORCE=1 to rebuild everything except the download)
	docker compose run --rm pipeline python -m reorder.pipeline all $(if $(FORCE),--force,)

test:
	python -m pytest

lint:
	ruff check src tests

demo:          ## call the API for a random held-out user
	@uid=$$(curl -s "localhost:8000/users/sample?n=1" | python -c "import sys,json;print(json.load(sys.stdin)[0]['user_id'])"); \
	echo "user $$uid"; \
	curl -s -X POST localhost:8000/predict -H 'content-type: application/json' \
	  -d "{\"user_id\": $$uid, \"k\": 10}" | python -m json.tool

clean-db:      ## drop the Postgres volume
	docker compose down -v

artifacts:     ## copy trained models + metrics from the Docker volume to ./artifacts
	docker compose cp api:/app/artifacts/. ./artifacts/
