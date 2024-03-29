# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Install libgl1-mesa-glx
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y libglib2.0-0

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run app.py when the container launches
CMD ["python", "app.py"]
