# Use an official, lightweight Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables to improve Python's behavior in Docker
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first. This is a Docker best practice
# that leverages layer caching. The dependencies will only be re-installed
# if the requirements.txt file changes.
COPY requirements.txt .

# Install the Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code (everything in the 'app' sub-directory)
# into the container's 'app' directory.
COPY ./app /app/app

# Expose the port that the FastAPI application will run on
EXPOSE 8000

# The command to run the application when the container starts.
# It uses uvicorn to serve the FastAPI app located in app/main.py.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]