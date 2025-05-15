# Use an official Python runtime as a parent image
FROM python:3.12-slim

ARG POSTGRESQL_DATABASE_URL
ARG BACKEND_API_URL
ARG USE_MOCK_BACKEND_API_URL
ARG MOCK_BACKEND_API_URL

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV POSTGRESQL_DATABASE_URL=$POSTGRESQL_DATABASE_URL
ENV BACKEND_API_URL=$BACKEND_API_URL
ENV USE_MOCK_BACKEND_API_URL=$USE_MOCK_BACKEND_API_URL
ENV MOCK_BACKEND_API_URL=$MOCK_BACKEND_API_URL

# Set the working directory in the container to /home
WORKDIR /home

# Create apa directory inside /home
RUN mkdir /home/app

# Add the module directory contents into the container at /home/app
ADD ./app /home/app

RUN ls -R /home

# Copy the requirements file into the container at /home
ADD ./requirements.txt /home

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run app.py when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
