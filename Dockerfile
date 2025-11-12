# Use official Python image
FROM python:3.10

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of your code (without ignored files/folders)
COPY . .

# Default command to run your Python app
# (change script.py to your main Python file)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
