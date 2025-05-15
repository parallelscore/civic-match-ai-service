#!/bin/bash

# Run the tests with coverage
coverage run --source=. -m pytest tests/ -vv

# Generate an HTML report
coverage html

# Generate a console report
coverage report

# Open the HTML report
open htmlcov/index.html