setup:
	pip install -r requirements.txt

setup-dev:
	pip install -r requirements.txt -r requirements_dev.txt

build-synthetic-data:
	python scripts/friedman_dataset.py


# -------------------------
# Individual presets
# -------------------------

run-preset-1:
	python src/main.py all -p preset_1 -n 1 -vv

run-preset-2:
	python src/main.py all -p preset_2 -n 1 -vv

run-preset-3:
	python src/main.py all -p preset_3 -n 1 -vv

run-preset-4:
	python src/main.py all -p preset_4 -n 1 -vv

run-preset-5:
	python src/main.py all -p preset_5 -n 1 -vv

run-preset-6:
	python src/main.py all -p preset_6 -n 1 -vv
	
plot-figures:
	Rscript src/figures/plot_figures.R

# -------------------------
# Pipeline (IMPORTANT PART)
# -------------------------

run-all-presets: run-preset-1 run-preset-2 run-preset-3 run-preset-4 run-preset-5 run-preset-6

plot-figures:
	Rscript src/figures/plot_figures.R

run-pipeline: run-all-presets plot-figures
