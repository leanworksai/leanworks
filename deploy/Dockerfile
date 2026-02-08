# Multi-stage build to reduce final image size
FROM python:3.10-slim as builder

WORKDIR /build

# Install only build dependencies (git needed for pip installs from git repos)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment and install dependencies into it
COPY requirements.txt .
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy application code and install leanworks package into venv
COPY --chown=root:root . .
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps . && \
    find /opt/venv -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete

# Final stage - minimal runtime image
FROM python:3.10-slim

WORKDIR /app

# No system dependencies needed at runtime (removed jq, curl, git)
RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user with home directory for library caches
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Copy virtual environment from builder (all packages + leanworks installed)
COPY --from=builder /opt/venv /opt/venv

# Copy only necessary application files (exclude tests, docs, etc.)
COPY --chown=appuser:appuser leanworks/ /app/leanworks/
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser run.py /app/
COPY --chown=appuser:appuser setup.py /app/
COPY --chown=appuser:appuser asgi.py /app/

# Clean up any remaining cache and temporary files
RUN find /opt/venv -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true && \
    find /opt/venv -type f -name "*.pyc" -delete && \
    find /opt/venv -type f -name "*.pyo" -delete && \
    rm -rf /root/.cache /tmp/*

# Switch to non-root user
USER appuser

# Set HOME and use venv Python by default
ENV HOME="/home/appuser"
ENV PATH="/opt/venv/bin:${PATH}"

EXPOSE 8000
