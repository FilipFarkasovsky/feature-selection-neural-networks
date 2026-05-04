setup:
	pip install -r requirements.txt

setup-dev:
	pip install -r requirements.txt -r requirements_dev.txt

lint:
	flake8

build-synthetic-data:
	python scripts/synthetic_datasets.py

build-xor-data:
	python scripts/xor_dataset.py

download-cumida-data:
	python scripts/download_cumida_datasets.py

run:
	python src/main.py all -p default -n 1 -vv

run-test-algorithms:
	python src/main.py all -p test_algorithms -n 1 -vv

run-test-pipeline:
	python src/main.py all -p test_pipeline -n 1 -vv

run-reduced:
	python src/main.py all -p reduced -n 31 -vv

run-test:
	python src/main.py all -p test_preset -n 1 -vv

run-preset-1:
	python src/main.py all -p preset_1 -n 1 -vv

run-preset-2:
	python src/main.py all -p preset_2 -n 1 -vv

run-preset-3:
	python src/main.py all -p preset_3 -n 1 -vv

run-preset-4:
	python src/main.py all -p preset_4 -n 1 -vv
