# Multi-stage build to reduce final image size
FROM python:3.10-slim as builder

WORKDIR /build

# Install only build dependencies (git needed for pip installs from git repos)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --prefix=/install -r requirements.txt

# Copy application code and install leanworks package
COPY --chown=root:root . .
RUN pip3 install --no-cache-dir --prefix=/install . && \
    find /install -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true && \
    find /install -type f -name "*.pyc" -delete && \
    find /install -type f -name "*.pyo" -delete

# Final stage - minimal runtime image
FROM python:3.10-slim

WORKDIR /app

# No system dependencies needed at runtime (removed jq, curl, git)
# Only keep minimal runtime dependencies if needed
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user with home directory for library caches
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only necessary application files (exclude tests, docs, etc.)
COPY --chown=appuser:appuser leanworks/ /app/leanworks/
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser run.py /app/
COPY --chown=appuser:appuser setup.py /app/
COPY --chown=appuser:appuser asgi.py /app/

# Clean up any remaining cache and temporary files
RUN find /usr/local -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true && \
    find /usr/local -type f -name "*.pyc" -delete && \
    find /usr/local -type f -name "*.pyo" -delete && \
    rm -rf /root/.cache /tmp/*

# Switch to non-root user
USER appuser

# Set HOME environment variable explicitly
ENV HOME="/home/appuser"
ENV PATH="/usr/local/bin:${PATH}"

EXPOSE 8000
