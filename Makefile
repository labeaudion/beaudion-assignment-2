# Makefile for Flask K-Means Clustering Project

# Python environment setup
PYTHON=python3
PIP=pip3

# List of required packages
REQUIREMENTS=requirements.txt

# Default target
all: install

# Install dependencies
install:
	$(PIP) install -r $(REQUIREMENTS)

# Run the application
run:
	$(PYTHON) app.py

# Clean up any temporary files
clean:
	find . -type __pycache__ -exec rm -rf {} \;

.PHONY: all install run clean
