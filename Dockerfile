FROM python:3.10

# Set environment variables for better memory management
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=2
ENV NUMEXPR_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=2

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
COPY requirements.api.txt .

# Install PaddlePaddle first to avoid conflicts
RUN pip install paddlepaddle==3.1.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# Install other requirements
RUN pip install --no-cache-dir -r requirements.api.txt

# Copy the application code
COPY . .

# Add ulimit settings for the container
RUN echo "* soft nofile 65536" >> /etc/security/limits.conf && \
    echo "* hard nofile 65536" >> /etc/security/limits.conf



# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"] 