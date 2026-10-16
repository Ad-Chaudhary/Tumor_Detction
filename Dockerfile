# Use an official lightweight Python image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to prevent resolution bugs
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU version explicitly first
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy the requirements file and install remaining python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code and weights into the container
COPY . /app

# Expose the port that Streamlit runs on
EXPOSE 8501

# Command to run your Streamlit application when the container starts
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]