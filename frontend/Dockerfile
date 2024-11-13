# Use the official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install required Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Start the Flask application
CMD ["python", "app.py"]
