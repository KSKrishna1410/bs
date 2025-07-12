FROM python:3.10

# Install native dependencies needed by cv2 and other packages
RUN apt-get update && \
    apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgtk-3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install PaddlePaddle first to avoid conflicts
RUN pip install paddlepaddle==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Install other requirements
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p comprehensive_output && \
    mkdir -p bank_statement_output && \
    mkdir -p bs_reconstructed_pdfs && \
    mkdir -p ocr_outputs_reconstructed

# Expose the API port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/health || exit 1

# Start the API server
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8888", "--log-level", "info"] 