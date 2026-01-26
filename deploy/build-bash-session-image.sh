#!/bin/bash
set -e

PROJECT_ID="leanworks-474204"
REGION="us-west1"
IMAGE_NAME="leanworks-bash-session"
REGISTRY="$REGION-docker.pkg.dev/$PROJECT_ID/leanworks-docker-images"

echo "Building bash session image..."
docker build -f deploy/Dockerfile.bash-session -t $IMAGE_NAME:latest .

echo "Tagging for Artifact Registry..."
docker tag $IMAGE_NAME:latest $REGISTRY/$IMAGE_NAME:latest

echo "Pushing to Artifact Registry..."
docker push $REGISTRY/$IMAGE_NAME:latest

echo "✓ Image pushed: $REGISTRY/$IMAGE_NAME:latest"
