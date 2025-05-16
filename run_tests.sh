#!/bin/bash

# Run the tests with coverage
coverage run --source=. --omit="*/tests/*,*/migrations/*,*/__pycache__/*,*/venv/*,t_matching_engine.py,t_matching_engine_2.py" -m pytest tests/ -vv

# Generate an HTML report
coverage html

# Generate a console report
coverage report

# Open the HTML report
open htmlcov/index.html