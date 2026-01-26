#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Deployment script for Leanworks API
# Uses Google Cloud Build to build and push Docker images to Artifact Registry
# Deploys to Google Kubernetes Engine (GKE) Autopilot cluster
# Multi-tenant: Single deployment serves all clients (client determined at runtime)
#
# Usage: ./deploy.sh
# Example: ./deploy.sh

# Load environment variables from .env file (optional)
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# ---------------------------
# Authenticate with Service Account
# ---------------------------

# Use service account from gcp_credential.json for all operations
if [ ! -f "gcp_credential.json" ]; then
    echo "Error: gcp_credential.json not found."
    echo "Please ensure gcp_credential.json exists in the project root."
    exit 1
fi

# Authenticate using the service account
echo "Authenticating with service account from gcp_credential.json..."
gcloud auth activate-service-account --key-file=gcp_credential.json --quiet

# Get project ID from credentials
PROJECT_ID=$(jq -r '.project_id' gcp_credential.json)
DEPLOYMENT_SA=$(jq -r '.client_email' gcp_credential.json)

echo "Using service account: $DEPLOYMENT_SA"
echo "Project ID: $PROJECT_ID"

# Set the project
gcloud config set project "$PROJECT_ID"

# ---------------------------
# Configuration Variables
# ---------------------------

# PROJECT_ID and DEPLOYMENT_SA are already set from authentication step above

# GKE Cluster configuration
CLUSTER_NAME="leanworks-prod"
CLUSTER_REGION="us-west1" # Use REGION for regional clusters
CLUSTER_ZONE="" # Use ZONE for zonal clusters (leave empty if using region)

# Shared infrastructure (multi-tenant)
# Static IP and domain are shared across all clients
# Using the same IP and certificate as leanworks-hub
STATIC_IP_NAME="leanworks-hub-ip"
CLIENT_DOMAIN="leanworks.ai"  # Shared domain for all clients

# Container Registry Configuration
# Uses Artifact Registry (required for Cloud Build)
REGISTRY="us-west1-docker.pkg.dev"
REPO_NAME="leanworks-docker-images"

# Docker Image configuration
IMAGE_NAME="ask-api"

IMAGE_TAG=$(date +%Y%m%d-%H%M%S) # Use timestamp for unique tags
FULL_IMAGE_PATH="$REGISTRY/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME:$IMAGE_TAG"

# Path to deployment directory
DEPLOY_DIR="deploy"

# Kubernetes Deployment/Service YAML files
DEPLOYMENT_YAML="$DEPLOY_DIR/deployment.yaml"
SERVICE_YAML="$DEPLOY_DIR/service.yaml"
# Note: Using shared certificate from leanworks-hub (leanworks-hub-ssl-cert)
# Do not create a separate certificate - the ingress references the shared one
MANAGED_CERTIFICATE_YAML=""
INGRESS_YAML="$DEPLOY_DIR/consolidated_ingress.yaml"
BACKEND_CONFIG_YAML="$DEPLOY_DIR/backend-config.yaml"
FIREWALL_RULE_YAML="$DEPLOY_DIR/firewall-rule.yaml"

# Dockerfile paths
DOCKERFILE_DIR="$DEPLOY_DIR"
DOCKERFILE="$DEPLOY_DIR/Dockerfile"

# ---------------------------
# Function Definitions
# ---------------------------

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to create static IP address if it doesn't exist
create_static_ip() {
    echo "Checking if static IP address $STATIC_IP_NAME exists..."
    if ! gcloud compute addresses describe "$STATIC_IP_NAME" --global 2>/dev/null; then
        echo "Creating static IP address: $STATIC_IP_NAME"
        gcloud compute addresses create "$STATIC_IP_NAME" --global
        
        # Get the IP address that was just created
        STATIC_IP=$(gcloud compute addresses describe "$STATIC_IP_NAME" --global --format="value(address)")
        echo "Static IP created: $STATIC_IP"
        
        # Update the ingress YAML with the new static IP name
        if [[ -n "$INGRESS_YAML" && -f "$INGRESS_YAML" ]]; then
            echo "Updating ingress config with new static IP name..."
            # Backup the original YAML
            cp "$INGRESS_YAML" "${INGRESS_YAML}.bak"
            # Replace the static IP name in the ingress YAML
            sed -i "" "s/kubernetes.io\/ingress.global-static-ip-name: \".*\"/kubernetes.io\/ingress.global-static-ip-name: \"$STATIC_IP_NAME\"/" "$INGRESS_YAML"
            # Clean up backup file after the operation
            rm "${INGRESS_YAML}.bak"
        fi
    else
        STATIC_IP=$(gcloud compute addresses describe "$STATIC_IP_NAME" --global --format="value(address)")
        echo "Static IP $STATIC_IP_NAME already exists: $STATIC_IP"
    fi
}

# Function to update ingress host with shared domain
update_ingress_host() {
    if [[ -n "$INGRESS_YAML" && -f "$INGRESS_YAML" ]]; then
        echo "Using shared domain: $CLIENT_DOMAIN"
        # Ingress YAML already has the correct domain configured
        # No need to update if it matches
        if ! grep -q "host: \"$CLIENT_DOMAIN\"" "$INGRESS_YAML"; then
            echo "Updating ingress host with shared domain: $CLIENT_DOMAIN"
            cp "$INGRESS_YAML" "${INGRESS_YAML}.bak.domain"
            sed -i "" "s/host: \".*\"/host: \"$CLIENT_DOMAIN\"/" "$INGRESS_YAML"
            echo "Ingress host updated to $CLIENT_DOMAIN"
        fi
    fi
}

# Function to create Artifact Registry repository (if using Artifact Registry)
create_artifact_repo() {
    echo "Checking if Artifact Registry repository exists..."
    if ! gcloud artifacts repositories describe "$REPO_NAME" --location="$CLUSTER_REGION" >/dev/null 2>&1; then
        echo "Creating Artifact Registry repository: $REPO_NAME"
        gcloud artifacts repositories create "$REPO_NAME" \
            --repository-format=docker \
            --location="$CLUSTER_REGION" \
            --description="Docker repository for $IMAGE_NAME"
    else
        echo "Artifact Registry repository $REPO_NAME already exists."
    fi
}

# Function to create Autopilot GKE cluster
create_gke_cluster() {
    echo "Creating GKE Autopilot cluster..."

    if [ -n "$CLUSTER_REGION" ]; then
        # Regional Autopilot cluster
        gcloud container clusters create-auto "$CLUSTER_NAME" \
            --region "$CLUSTER_REGION" \
            --project "$PROJECT_ID"
    elif [ -n "$CLUSTER_ZONE" ]; then
        # Zonal Autopilot cluster (if supported)
        gcloud container clusters create-auto "$CLUSTER_NAME" \
            --zone "$CLUSTER_ZONE" \
            --project "$PROJECT_ID"
    else
        echo "Error: Neither CLUSTER_REGION nor CLUSTER_ZONE is set."
        exit 1
    fi

    echo "GKE Autopilot cluster '$CLUSTER_NAME' created successfully."
}

# Function to configure kubectl
configure_kubectl() {
    echo "Configuring kubectl to use the new cluster..."
    if [ -n "$CLUSTER_REGION" ]; then
        gcloud container clusters get-credentials "$CLUSTER_NAME" --region "$CLUSTER_REGION"
    elif [ -n "$CLUSTER_ZONE" ]; then
        gcloud container clusters get-credentials "$CLUSTER_NAME" --zone "$CLUSTER_ZONE"
    else
        echo "Error: Neither CLUSTER_REGION nor CLUSTER_ZONE is set."
        exit 1
    fi
}

# Function to enable required APIs
enable_required_apis() {
    echo "Enabling required Google Cloud APIs..."
    
    REQUIRED_APIS=(
        "cloudbuild.googleapis.com"
        "secretmanager.googleapis.com"
        "artifactregistry.googleapis.com"
        "container.googleapis.com"
    )
    
    for api in "${REQUIRED_APIS[@]}"; do
        echo "Enabling $api..."
        gcloud services enable "$api" --project="$PROJECT_ID"
    done
    
    echo "All required APIs enabled."
}

# Function to build and push Docker image using Cloud Build (default method)
build_and_push_cloud_build() {
    echo "Building and pushing Docker image using Cloud Build..."
    echo "No GitHub token needed since leanworks package is in the same repository..."
    
    echo "Submitting build to Cloud Build..."
    echo "You will see real-time build progress below..."
    
    # Submit build to Cloud Build
    # Format service account for Cloud Build (needs full resource path)
    CLOUDBUILD_SA="projects/$PROJECT_ID/serviceAccounts/$DEPLOYMENT_SA"
    
    if gcloud beta builds submit \
        --config cloudbuild.yaml \
        --substitutions _IMAGE_TAG="$IMAGE_TAG",_IMAGE_NAME="$IMAGE_NAME",_REGISTRY="$REGISTRY",_REPO_NAME="$REPO_NAME" \
        --service-account="$CLOUDBUILD_SA" \
        --project="$PROJECT_ID" \
        --region="$CLUSTER_REGION"; then
        echo "Cloud Build completed successfully!"
        echo "Image built and pushed: $FULL_IMAGE_PATH"
    else
        echo "Cloud Build failed. Check the logs above for details."
        exit 1
    fi
}

# Function to apply Bash Session RBAC (required for Kubernetes backend)
apply_bash_session_rbac() {
    RBAC_YAML="$DEPLOY_DIR/bash-session-rbac.yaml"
    
    if [[ ! -f "$RBAC_YAML" ]]; then
        echo "Warning: $RBAC_YAML not found. Skipping RBAC setup."
        return
    fi
    
    echo "Applying Bash Session RBAC (ServiceAccount, Role, RoleBinding)..."
    if kubectl apply -f "$RBAC_YAML"; then
        echo "✓ Bash Session RBAC applied successfully"
        echo "  - ServiceAccount: ask-api-sa"
        echo "  - Role: bash-session-manager"
        echo "  - Permissions: pods, pods/exec, persistentvolumeclaims"
    else
        echo "Error: Failed to apply Bash Session RBAC"
        exit 1
    fi
}

# Function to apply Kubernetes YAML
apply_kubernetes_yaml() {
    echo "Updating deployment YAML with the new image..."
    # Backup the original YAML
    cp "$DEPLOYMENT_YAML" "${DEPLOYMENT_YAML}.bak"
    if [[ -n "$SERVICE_YAML" ]]; then
        cp "$SERVICE_YAML" "${SERVICE_YAML}.bak"
    fi

    # Replace the image field with the full image path
    sed -i "" "s|image: .*|image: $FULL_IMAGE_PATH|" "$DEPLOYMENT_YAML"
    if [[ -n "$SERVICE_YAML" ]]; then
        sed -i "" "s|image: .*|image: $FULL_IMAGE_PATH|" "$SERVICE_YAML"
    fi

    # Try to delete existing pods if they exist
    echo "Trying to delete existing pods..."
    kubectl delete pods -l app=$(kubectl get deployments -o jsonpath='{.items[0].metadata.labels.app}' 2>/dev/null) 2>/dev/null || true

    echo "Applying Kubernetes YAML..."
    kubectl apply -f "$DEPLOYMENT_YAML"
    if [[ -n "$SERVICE_YAML" ]]; then
        kubectl apply -f "$SERVICE_YAML"
    fi
    
    # Apply ingress and backend config
    # Note: SSL certificate is managed by leanworks-hub (leanworks-hub-ssl-cert)
    # The ingress references the shared certificate, so we don't apply a separate one
    echo "Applying consolidated ingress and backend config..."
    if [[ -n "$BACKEND_CONFIG_YAML" ]]; then
        kubectl apply -f "$BACKEND_CONFIG_YAML"
    fi
    
    if [[ -n "$INGRESS_YAML" ]]; then
        kubectl apply -f "$INGRESS_YAML"
    fi
    
    # Certificate is shared from leanworks-hub, no need to apply separately
    if [[ -n "$MANAGED_CERTIFICATE_YAML" ]]; then
        echo "Note: Using shared SSL certificate from leanworks-hub (leanworks-hub-ssl-cert)"
        echo "Skipping managed-certificate.yaml - certificate is managed by leanworks-hub deployment"
    fi
    
    # Verify firewall rule exists (assumed to be created during initial infrastructure setup)
    # The firewall rule 'allow-proxy-traffic' is shared with leanworks-hub and should already exist
    if [[ -f "$FIREWALL_RULE_YAML" ]]; then
        echo "Verifying firewall rule exists..."
        if gcloud compute firewall-rules describe allow-proxy-traffic --project="$PROJECT_ID" &>/dev/null; then
            echo "✓ Firewall rule 'allow-proxy-traffic' exists. Proceeding with deployment."
        else
            echo "Warning: Firewall rule 'allow-proxy-traffic' not found."
            echo "The firewall rule should be created manually by a project administrator if it doesn't exist."
            echo "Deployment will continue, but ingress may not work without the firewall rule."
        fi
    fi
        
    echo "Note: SSL certificate provisioning may take up to 30 minutes"

    echo "Deployment applied successfully."

    # Remove the backup YAML files
    rm "${DEPLOYMENT_YAML}.bak"
    if [[ -n "$SERVICE_YAML" ]]; then
        rm "${SERVICE_YAML}.bak"
    fi
}


# ---------------------------
# Main Script Execution
# ---------------------------

# Check if required commands are available
for cmd in gcloud kubectl jq yq; do
    if ! command_exists "$cmd"; then
        echo "Error: $cmd is not installed. Please install it before running this script."
        exit 1
    fi
done

# Enable required Google Cloud APIs
enable_required_apis

# Create Artifact Registry repository (if using Artifact Registry)
if [ "$REGISTRY" != "gcr.io" ]; then
    create_artifact_repo
fi

# Create static IP address if needed (shared for all clients)
create_static_ip

# Verify ingress host matches shared domain
update_ingress_host

# Check if GKE cluster exists
echo "Checking if GKE cluster $CLUSTER_NAME exists..."
if ! gcloud container clusters list --filter="name:$CLUSTER_NAME" --format="value(name)" | grep -q "$CLUSTER_NAME"; then
    echo "GKE cluster $CLUSTER_NAME does not exist. Creating..."
    create_gke_cluster
else
    echo "GKE cluster $CLUSTER_NAME already exists."
fi

# Configure kubectl
configure_kubectl

# Apply Bash Session RBAC (required for Kubernetes backend to work)
apply_bash_session_rbac

# Build and push Docker image using Cloud Build (default method)
build_and_push_cloud_build

# Apply Kubernetes YAML
apply_kubernetes_yaml

# ---------------------------
# End of Script
# ---------------------------

# Clean up any remaining backup files
echo "Cleaning up temporary backup files..."
if [[ -n "$INGRESS_YAML" ]]; then
    rm -f "${INGRESS_YAML}.bak.domain" "${INGRESS_YAML}.bak.cert"
fi

echo "Deployment of API to GKE completed successfully!"
echo "Multi-tenant API service deployed with public endpoints and SSL certificate"
echo "Please allow up to 30 minutes for SSL certificate provisioning"
echo "Your service will be available at: https://$CLIENT_DOMAIN"
echo "Note: Client is determined at runtime from user_id"

