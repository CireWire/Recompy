# Use an official Python runtime as a parent image
FROM python:3.10.0a6-alpine3.13

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Expose port 80 for the Flask app
EXPOSE 80

# Define environment variable
ENV NAME Recommender

# Run recompy.py when the container launches
CMD ["python", "recompy.py"]
