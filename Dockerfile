# Use the official Python image as the base image
FROM python:3.8

WORKDIR /app

COPY . /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

# Run the FastAPI server when the container starts
CMD gunicorn --bind 0.0.0.0:$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker --threads 8 --timeout 0 main:app
