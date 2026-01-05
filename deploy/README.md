# Deployment Guide

## GCP Credentials Setup

The application requires GCP credentials to access Secret Manager and other GCP services. These credentials are mounted as a Kubernetes Secret.

### Creating the Kubernetes Secret

1. Ensure you have `gcp_credential.json` in the project root directory
2. Run the script to create the secret:

```bash
cd deploy
./create-gcp-credentials-secret.sh [namespace]
```

If no namespace is specified, it defaults to `default`.

Example:
```bash
./create-gcp-credentials-secret.sh production
```

### Manual Secret Creation

Alternatively, you can create the secret manually:

```bash
kubectl create secret generic gcp-credentials \
    --from-file=gcp_credential.json \
    -n <namespace>
```

### Verifying the Secret

```bash
kubectl get secret gcp-credentials -n <namespace>
kubectl describe secret gcp-credentials -n <namespace>
```

### Updating the Secret

If you need to update the credentials:

```bash
# Delete the existing secret
kubectl delete secret gcp-credentials -n <namespace>

# Recreate it
./create-gcp-credentials-secret.sh <namespace>
```

Or update it directly:
```bash
kubectl create secret generic gcp-credentials \
    --from-file=gcp_credential.json \
    --dry-run=client -o yaml | kubectl apply -f - -n <namespace>
```

## Deployment

The deployment automatically mounts the secret at `/app/gcp_credential.json` in the container. The application will:

1. First check for `gcp_credential.json` at `/app/gcp_credential.json`
2. If not found, fall back to Application Default Credentials (ADC)
3. Use environment variables as a final fallback for specific secrets (DB_PASSWORD, API_KEY)

## Troubleshooting

### Secret not found
- Verify the secret exists: `kubectl get secret gcp-credentials -n <namespace>`
- Check the deployment volume mount configuration

### Permission errors
- Ensure the service account in `gcp_credential.json` has the necessary IAM roles:
  - `roles/secretmanager.secretAccessor` for Secret Manager access
  - `roles/cloudsql.client` for Cloud SQL access
  - `roles/datastore.user` for Firestore access

### Application still can't access secrets
- Check pod logs: `kubectl logs <pod-name> -n <namespace>`
- Verify the credential file is mounted: `kubectl exec <pod-name> -n <namespace> -- ls -la /app/gcp_credential.json`
- Check file permissions: `kubectl exec <pod-name> -n <namespace> -- cat /app/gcp_credential.json | head -5`

