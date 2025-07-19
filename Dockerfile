FROM python:3.10

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY bank_statements/requirements.txt bank_statements/requirements.txt
COPY api/requirements.txt api/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r bank_statements/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copy source code
COPY bank_statements/ bank_statements/
COPY api/ api/
COPY bankstmt_allkeys.csv .
COPY IFSC_master.csv .

# Install local package
RUN pip install -e bank_statements/

# Run API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"] 