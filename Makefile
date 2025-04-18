.PHONY: run setup clean test

# Default target
all: setup

# Run the Streamlit app
run:
	@echo "Starting Streamlit app..."
	streamlit run app.py

# Set up the virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment..."
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -r requirements-dev.txt

# Clean up temporary files
clean:
	@echo "Cleaning up..."
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage

# Run tests (placeholder for future use)
test:
	@echo "Running tests..."
	# Add test commands here when tests are implemented 