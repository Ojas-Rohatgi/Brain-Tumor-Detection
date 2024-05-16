# Use the official Python image as a base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Flask and other dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir flask tensorflow opencv-python-headless

# Expose port 5000 to the outside world
EXPOSE 5000

# Define environment variable
ENV FLASK_APP app.py

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
