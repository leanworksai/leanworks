#!/bin/bash

# Script to create Kubernetes secret for GCP credentials
# Usage: ./create-gcp-credentials-secret.sh [namespace]

set -e

NAMESPACE=${1:-default}
SECRET_NAME="gcp-credentials"
CREDENTIAL_FILE="gcp_credential.json"

if [ ! -f "$CREDENTIAL_FILE" ]; then
    echo "Error: $CREDENTIAL_FILE not found in current directory"
    echo "Please ensure the credential file exists before running this script"
    exit 1
fi

echo "Creating Kubernetes secret '$SECRET_NAME' in namespace '$NAMESPACE'..."

# Delete existing secret if it exists
kubectl delete secret "$SECRET_NAME" -n "$NAMESPACE" 2>/dev/null || true

# Create secret from file
kubectl create secret generic "$SECRET_NAME" \
    --from-file="$CREDENTIAL_FILE" \
    -n "$NAMESPACE"

echo "✅ Secret '$SECRET_NAME' created successfully in namespace '$NAMESPACE'"
echo ""
echo "To verify the secret was created:"
echo "  kubectl get secret $SECRET_NAME -n $NAMESPACE"
echo ""
echo "To view the secret (base64 encoded):"
echo "  kubectl get secret $SECRET_NAME -n $NAMESPACE -o yaml"

