VENV_DIR = .local
REQUIREMENTS_FILE = requirements.txt
ACTIVATE = . $(VENV_DIR)/bin/activate

PYTHONPATH ?= $(PWD)

export PYTHONUNBUFFERED = 1
export PYTHONPATH

$(VENV_DIR):
	python3 -m venv $(VENV_DIR)
	$(ACTIVATE) && pip install --upgrade pip && pip install -r $(REQUIREMENTS_FILE)

install: $(VENV_DIR)

update:
	$(ACTIVATE) && pip install --upgrade -r $(REQUIREMENTS_FILE)

clean:
	rm -rf $(VENV_DIR) \
	rm -rf recommendations

remove: clean
	rm -f $(REQUIREMENTS_FILE)

run:
	$(ACTIVATE) && python3 src/jobs/loaders/run.py

docker_build_nocache:
	docker build --progress=plain --no-cache -t nrec-worker .

docker_build:
	docker build -t nrec-worker .

docker_without_stop:
	docker run --rm --entrypoint /bin/sh nrec-worker -c "tail -f /dev/null"

docker_submit:
	docker run --rm \
		-e ENV=prod \
		-v ./:/app \
		--entrypoint /bin/sh \
		nrec-worker \
		-c "/opt/spark/bin/spark-submit \
		--conf spark.driver.extraClassPath=/opt/spark/lib-jars/* \
		--conf spark.executor.extraClassPath=/opt/spark/lib-jars/* \
		--conf spark.submit.pyFiles=/app/src/* \
		local:///app/src/runner.py"

format:
	$(ACTIVATE) && python3 -m black src


.PHONY: install update clean remove run_local run docker_build docker_without_stop docker_run docker_submit format
