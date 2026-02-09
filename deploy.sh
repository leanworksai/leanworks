#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Deployment script for Leanworks API
# Uses Google Cloud Build to build and push Docker images to Artifact Registry
# Deploys to Google Kubernetes Engine (GKE) Autopilot cluster
# Multi-tenant: Single deployment serves all clients (client determined at runtime)
#
# Prerequisites:
# - Service account with the following roles:
#   * Artifact Registry Administrator/Repository Administrator
#   * Cloud Build Service Account
#   * Kubernetes Engine Admin
#   * Secret Manager Admin
#   * Storage Admin
#   * Compute Network Admin (for static IP creation)
#   * Service Usage Admin (for enabling APIs)
#
# Usage: ./deploy.sh [environment]
# Example: ./deploy.sh dev
# Example: ./deploy.sh prod

# Load environment variables from .env file (optional)
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# ---------------------------
# Environment selection
# ---------------------------

ENVIRONMENT=${1:-prod}
if [ "$ENVIRONMENT" != "prod" ] && [ "$ENVIRONMENT" != "dev" ]; then
    echo "Invalid environment: $ENVIRONMENT"
    echo "Usage: ./deploy/deploy.sh [dev|prod]"
    exit 1
fi

if [ "$ENVIRONMENT" = "dev" ]; then
    CREDENTIAL_FILE="gcp_credential_dev.json"
    CLUSTER_NAME="leanworks-dev"
    STATIC_IP_NAME="leanworks-dev-hub-ip"
    CLIENT_DOMAIN="dev.leanworks.ai"
else
    CREDENTIAL_FILE="gcp_credential.json"
    CLUSTER_NAME="leanworks-prod"
    STATIC_IP_NAME="leanworks-hub-ip"
    CLIENT_DOMAIN="leanworks.ai"
fi

# ---------------------------
# Authenticate with Service Account
# ---------------------------

if [ ! -f "$CREDENTIAL_FILE" ]; then
    echo "Error: $CREDENTIAL_FILE not found."
    echo "Please ensure the credential file exists in the project root."
    exit 1
fi

# Authenticate using the service account
echo "Authenticating with service account from $CREDENTIAL_FILE..."
gcloud auth activate-service-account --key-file="$CREDENTIAL_FILE" --quiet

# Get project ID from credentials
PROJECT_ID=$(jq -r '.project_id' "$CREDENTIAL_FILE")
DEPLOYMENT_SA=$(jq -r '.client_email' "$CREDENTIAL_FILE")

echo "Using service account: $DEPLOYMENT_SA"
echo "Project ID: $PROJECT_ID"

# Set the project
gcloud config set project "$PROJECT_ID"

# ---------------------------
# Configuration Variables
# ---------------------------

# PROJECT_ID and DEPLOYMENT_SA are already set from authentication step above

# GKE Cluster configuration
CLUSTER_REGION="us-west1" # Use REGION for regional clusters
CLUSTER_ZONE="" # Use ZONE for zonal clusters (leave empty if using region)

# Shared infrastructure (multi-tenant)
# Static IP and domain are shared across all clients
# Using the same IP and certificate as leanworks-hub
# Static IP and domain are environment-specific

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
if [ "$ENVIRONMENT" = "dev" ]; then
    DEPLOYMENT_YAML="$DEPLOY_DIR/deployment-dev.yaml"
fi
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
    echo "Checking static IP address $STATIC_IP_NAME..."

    # Try to describe the IP to check if it exists
    if gcloud compute addresses describe "$STATIC_IP_NAME" --global --format="value(address)" 2>/dev/null; then
        STATIC_IP=$(gcloud compute addresses describe "$STATIC_IP_NAME" --global --format="value(address)")
        echo "✓ Static IP $STATIC_IP_NAME exists: $STATIC_IP"
        return 0
    fi

    # Check if the failure was due to permission denied vs IP not existing
    if gcloud compute addresses describe "$STATIC_IP_NAME" --global 2>&1 | grep -q "PERMISSION_DENIED\|permission"; then
        echo "⚠️ Cannot check static IP status due to insufficient permissions."
        echo "   Assuming $STATIC_IP_NAME exists (may have been created by leanworks-hub)."
        echo "   Continuing with deployment..."
        return 0
    fi

    # If we reach here, the IP doesn't exist and we have permissions, try to create it
    echo "Creating static IP address: $STATIC_IP_NAME"
    if gcloud compute addresses create "$STATIC_IP_NAME" --global 2>/dev/null; then
        # Get the IP address that was just created
        STATIC_IP=$(gcloud compute addresses describe "$STATIC_IP_NAME" --global --format="value(address)")
        echo "✓ Static IP created: $STATIC_IP"

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
        echo "❌ Failed to create static IP $STATIC_IP_NAME"
        echo "   This may be due to insufficient permissions or the IP already exists."
        echo "   The deployment may still work if the IP was pre-created by an administrator."
        echo "   Continuing with deployment..."
        return 1
    fi
}

# Function to update ingress host with shared domain
update_ingress_host() {
    if [[ -n "$INGRESS_YAML" && -f "$INGRESS_YAML" ]]; then
        if [ "$ENVIRONMENT" = "dev" ]; then
            DEV_DOMAIN="dev.leanworks.ai"
            echo "Using dev domain: $DEV_DOMAIN"
            # Verify dev domain is in ingress YAML
            if ! grep -q "host: \"$DEV_DOMAIN\"" "$INGRESS_YAML"; then
                echo "Warning: Dev domain $DEV_DOMAIN not found in ingress YAML"
            fi
        else
            CLIENT_DOMAIN="leanworks.ai"
            echo "Using prod domain: $CLIENT_DOMAIN"
        fi
    fi
}

# Function to update ingress certificate and static IP based on environment
update_ingress_ssl_config() {
    if [[ -n "$INGRESS_YAML" && -f "$INGRESS_YAML" ]]; then
        if [ "$ENVIRONMENT" = "dev" ]; then
            CERT_NAME="dev-leanworks-ai-ssl-cert"
            STATIC_IP_NAME="leanworks-dev-hub-ip"
            echo "Using dev SSL certificate: $CERT_NAME"
            echo "Using dev static IP: $STATIC_IP_NAME"
        else
            CERT_NAME="leanworks-hub-ssl-cert"
            STATIC_IP_NAME="leanworks-hub-ip"
            echo "Using prod SSL certificate: $CERT_NAME"
            echo "Using prod static IP: $STATIC_IP_NAME"
        fi

        # Backup the original YAML
        cp "$INGRESS_YAML" "${INGRESS_YAML}.bak.ssl"

        # Update certificate annotation
        sed -i "" "s/networking.gke.io\/managed-certificates: \".*\"/networking.gke.io\/managed-certificates: \"$CERT_NAME\"/" "$INGRESS_YAML"

        # Update static IP annotation
        sed -i "" "s/kubernetes.io\/ingress.global-static-ip-name: \".*\"/kubernetes.io\/ingress.global-static-ip-name: \"$STATIC_IP_NAME\"/" "$INGRESS_YAML"

        echo "Updated ingress SSL configuration for $ENVIRONMENT environment"
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
        if gcloud services enable "$api" --project="$PROJECT_ID" 2>/dev/null; then
            echo "✓ $api enabled successfully"
        else
            echo "⚠️ Failed to enable $api (may already be enabled or insufficient permissions)"
        fi
    done

    echo "API enable check completed."
}

# Function to build and push Docker image using Cloud Build (default method)
build_and_push_cloud_build() {
    echo "Building and pushing Docker image using Cloud Build..."
    echo "No GitHub token needed since leanworks package is in the same repository..."
    
    echo "Submitting build to Cloud Build..."
    echo "You will see real-time build progress below..."
    
    # Submit build to Cloud Build
    # Use deployment SA for Cloud Build (same account)
    if gcloud beta builds submit \
        --config deploy/cloudbuild.yaml \
        --substitutions _IMAGE_TAG="$IMAGE_TAG",_IMAGE_NAME="$IMAGE_NAME",_REGISTRY="$REGISTRY",_REPO_NAME="$REPO_NAME" \
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

# Function to build and push session manager image
build_session_manager_image() {
    echo ""
    echo "=========================================="
    echo "Building Bash Session Manager Docker Image"
    echo "=========================================="

    if [[ ! -f "deploy/Dockerfile.session-manager" ]]; then
        echo "Error: deploy/Dockerfile.session-manager not found."
        exit 1
    fi

    SESSION_IMAGE_NAME="bash-session-manager"
    SESSION_REGISTRY="$REGISTRY/$PROJECT_ID/$REPO_NAME/$SESSION_IMAGE_NAME"

    echo "Step 1: Setting up Docker buildx for cross-platform builds..."
    # Create and use a buildx builder that supports multi-platform builds
    docker buildx create --name leanworks-builder --use 2>/dev/null || docker buildx use leanworks-builder

    echo "Step 2: Building Docker image from deploy/Dockerfile.session-manager for linux/amd64..."
    # Build for linux/amd64 (x86_64) which is what GKE uses by default
    docker buildx build \
        --platform linux/amd64 \
        --file deploy/Dockerfile.session-manager \
        --tag $SESSION_REGISTRY:latest \
        --push \
        .
    if [ $? -ne 0 ]; then
        echo "Error: Docker buildx failed"
        exit 1
    fi
    echo "✓ Docker image built and pushed successfully for linux/amd64: $SESSION_REGISTRY:latest"

    echo ""
    echo "=========================================="
    echo "Session Manager image build completed!"
    echo "=========================================="
}

# Function to deploy session manager service
deploy_session_manager() {
    echo ""
    echo "=========================================="
    echo "Deploying Bash Session Manager Service"
    echo "=========================================="
    
    RBAC_YAML="$DEPLOY_DIR/bash-session-manager-rbac.yaml"
    DEPLOYMENT_YAML_SM="$DEPLOY_DIR/bash-session-manager-deployment.yaml"
    if [ "$ENVIRONMENT" = "dev" ]; then
        DEPLOYMENT_YAML_SM="$DEPLOY_DIR/bash-session-manager-deployment-dev.yaml"
    fi
    SERVICE_YAML_SM="$DEPLOY_DIR/bash-session-manager-service.yaml"
    CRONJOB_YAML="$DEPLOY_DIR/bash-session-cleanup-cronjob.yaml"
    
    # Check if all required files exist
    for yaml_file in "$RBAC_YAML" "$DEPLOYMENT_YAML_SM" "$SERVICE_YAML_SM" "$CRONJOB_YAML"; do
        if [[ ! -f "$yaml_file" ]]; then
            echo "Warning: $yaml_file not found. Skipping session manager deployment."
            return
        fi
    done
    
    echo "Step 1: Applying Session Manager RBAC..."
    if kubectl apply -f "$RBAC_YAML"; then
        echo "✓ Session Manager RBAC applied successfully"
        echo "  - ServiceAccount: bash-session-manager-sa"
        echo "  - Role: bash-session-manager-role"
    else
        echo "Error: Failed to apply Session Manager RBAC"
        exit 1
    fi
    
    echo ""
    echo "Step 2: Deploying Session Manager (2 replicas for HA)..."
    if kubectl apply -f "$DEPLOYMENT_YAML_SM"; then
        echo "✓ Session Manager deployment applied successfully"
    else
        echo "Error: Failed to apply Session Manager deployment"
        exit 1
    fi
    
    echo ""
    echo "Step 3: Creating Session Manager Service..."
    if kubectl apply -f "$SERVICE_YAML_SM"; then
        echo "✓ Session Manager service created successfully"
    else
        echo "Error: Failed to create Session Manager service"
        exit 1
    fi
    
    echo ""
    echo "Step 4: Setting up Cleanup CronJob (runs every hour)..."
    if kubectl apply -f "$CRONJOB_YAML"; then
        echo "✓ Cleanup CronJob created successfully"
    else
        echo "Error: Failed to create Cleanup CronJob"
        exit 1
    fi
    
    echo ""
    echo "Waiting for Session Manager pods to be ready (max 60 seconds)..."
    if kubectl rollout status deployment/bash-session-manager --timeout=60s 2>/dev/null; then
        echo "✓ Session Manager pods are ready"
        
        # Get pod count
        POD_COUNT=$(kubectl get pods -l app=bash-session-manager --no-headers | wc -l)
        echo "  - Active pods: $POD_COUNT"
        
        # Verify service connectivity
        echo ""
        echo "Verifying Session Manager service..."
        SERVICE_IP=$(kubectl get svc bash-session-manager-service -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
        if [[ -n "$SERVICE_IP" ]]; then
            echo "✓ Session Manager service is available at: $SERVICE_IP:8080"
        fi
    else
        echo "Warning: Session Manager pods did not become ready within timeout"
        echo "Check pod logs with: kubectl logs -f deployment/bash-session-manager"
    fi
    
    echo ""
    echo "=========================================="
    echo "Session Manager deployment completed!"
    echo "=========================================="
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
echo "Checking required tools..."
for cmd in gcloud kubectl jq yq; do
    if ! command_exists "$cmd"; then
        echo "❌ Error: $cmd is not installed. Please install it before running this script."
        exit 1
    fi
done
echo "✓ All required tools are available"

# Enable required Google Cloud APIs (may fail if insufficient permissions)
echo ""
enable_required_apis

# Create Artifact Registry repository (if using Artifact Registry)
if [ "$REGISTRY" != "gcr.io" ]; then
    create_artifact_repo
fi

# Create static IP address if needed (shared for all clients)
create_static_ip

# Verify ingress host matches shared domain
update_ingress_host

# Update ingress SSL configuration for environment
update_ingress_ssl_config

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

# Build and push Session Manager image first (before deploying pods that depend on it)
build_session_manager_image

# Deploy Session Manager Kubernetes resources
deploy_session_manager

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
echo ""
echo "=========================================="
echo "Deployment Summary"
echo "=========================================="
echo "✓ Bash Session Manager Service deployed (2 replicas)"
echo "  - Handles bash session management"
echo "  - 2-replica HA deployment"
echo "  - Automatic hourly cleanup via CronJob"
echo ""
echo "✓ Multi-tenant API service deployed"
echo "  - Public endpoints with SSL certificate"
echo "  - Integrated with Session Manager"
echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Wait up to 30 minutes for SSL certificate provisioning"
echo "2. Service will be available at: https://$CLIENT_DOMAIN"
echo "3. Check Session Manager status:"
echo "   kubectl get pods -l app=bash-session-manager"
echo "4. Monitor Session Manager logs:"
echo "   kubectl logs -f deployment/bash-session-manager"
echo "5. Test Session Manager endpoints:"
echo "   kubectl port-forward svc/bash-session-manager-service 8080:8080"
echo "   curl http://localhost:8080/health"
echo ""
echo "Note: Client is determined at runtime from user_id"
echo "=========================================="
