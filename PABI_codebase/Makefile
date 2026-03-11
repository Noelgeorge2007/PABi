# Makefile for PABI QSPR Analysis Pipeline
# Usage: make all

PYTHON = python3
ALPHA = 0.75

.PHONY: all dataset qspr diagnostics figures drugs alpha clean help

all: dataset qspr figures
	@echo "Pipeline complete. Check results/ and figures/ directories."

dataset:
	$(PYTHON) prepare_dataset.py --alpha $(ALPHA)

qspr: dataset
	$(PYTHON) qspr_models.py --data data/dataset_250.csv --alpha $(ALPHA)

diagnostics: qspr
	$(PYTHON) main.py --step diagnostics --alpha $(ALPHA)

figures: qspr
	$(PYTHON) figures.py --data data/dataset_250.csv

drugs:
	$(PYTHON) drug_analysis.py

alpha: dataset
	$(PYTHON) main.py --step alpha --alpha $(ALPHA)

full:
	$(PYTHON) main.py --step all --alpha $(ALPHA)

clean:
	rm -rf data/ results/ figures/
	rm -rf __pycache__/ *.pyc

help:
	@echo "PABI QSPR Analysis Pipeline"
	@echo ""
	@echo "Targets:"
	@echo "  all          Run dataset + qspr + figures (default)"
	@echo "  dataset      Prepare compound dataset"
	@echo "  qspr         Run QSPR regression models"
	@echo "  diagnostics  Run regression diagnostics"
	@echo "  figures       Generate publication figures"
	@echo "  drugs        FDA drug analysis"
	@echo "  alpha        Alpha optimization"
	@echo "  full         Run complete pipeline (all steps)"
	@echo "  clean        Remove generated files"
	@echo ""
	@echo "Variables:"
	@echo "  ALPHA=0.75   Scaling exponent (default: 0.75)"
