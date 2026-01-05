FROM python:3.10-slim
WORKDIR /app

# Install system dependencies first
RUN apt-get update && apt-get install -y git jq curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user with home directory
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

# Copy just requirements.txt first to leverage Docker cache
COPY requirements.txt .

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add user's local bin to PATH so hypercorn can be found
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Install Python dependencies
RUN echo "Installing base requirements..." && \
    pip3 install --no-cache-dir --user -r requirements.txt && \
    echo "All dependencies installed successfully"

# Copy the rest of the application code (includes leanworks package)
COPY --chown=appuser:appuser . .

# Install leanworks package locally (in editable mode since it's in the same repo)
RUN echo "Installing leanworks package locally..." && \
    pip3 install --no-cache-dir --user -e . && \
    echo "Verifying leanworks installation..." && \
    python3 -c "import leanworks; print(f'leanworks installed at: {leanworks.__file__}')" || (echo "ERROR: leanworks package installation verification failed" && exit 1) && \
    echo "leanworks package installed successfully"

EXPOSE 8000
