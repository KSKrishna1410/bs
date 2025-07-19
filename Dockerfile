FROM python:3.10

# Set environment variables for better memory management
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2
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

# Add ulimit settings for the container
RUN echo "* soft nofile 65536" >> /etc/security/limits.conf && \
    echo "* hard nofile 65536" >> /etc/security/limits.conf

# Add DNS settings
RUN echo "nameserver 8.8.8.8" > /etc/resolv.conf && \
    echo "nameserver 8.8.4.4" >> /etc/resolv.conf

WORKDIR /app

# Copy the application code
COPY . .

# Install PaddlePaddle first to avoid conflicts
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pip -U && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ paddlepaddle==3.1.0 || \
    (sleep 5 && pip install --no-cache-dir -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ paddlepaddle==3.1.0) || \
    (sleep 10 && pip install --no-cache-dir -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ paddlepaddle==3.1.0)

# Install other requirements
RUN pip install --no-cache-dir -r requirements.api.txt

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"] 