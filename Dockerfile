# Use a base Python image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy required files (excluding venv via .dockerignore)
COPY requirements.txt ./
COPY app.py ./
COPY models/ ./models/
COPY templates/ ./templates/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
