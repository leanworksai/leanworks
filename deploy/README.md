# Deployment Guide

## GCP Credentials Setup

The application requires GCP credentials to access Secret Manager and other GCP services. These credentials are mounted as a Kubernetes Secret.

### Creating the Kubernetes Secret

1. Ensure you have the correct credential file in the project root directory:
   - Prod: `gcp_credential.json`
   - Dev: `gcp_credential_dev.json`
2. Run the script to create the secret:

```bash
cd deploy
./create-gcp-credentials-secret.sh [environment] [namespace]
```

If no environment is specified, it defaults to `prod`. If no namespace is specified, it defaults to `default`.

Example:
```bash
./create-gcp-credentials-secret.sh prod default
./create-gcp-credentials-secret.sh dev default
```

### Manual Secret Creation

Alternatively, you can create the secret manually:

```bash
kubectl create secret generic gcp-credentials \
    --from-file=gcp_credential.json \
    -n <namespace>

# Dev secret
kubectl create secret generic gcp-credentials-dev \
    --from-file=gcp_credential_dev.json \
    -n <namespace>
```

### Verifying the Secret

```bash
kubectl get secret gcp-credentials -n <namespace>
kubectl describe secret gcp-credentials -n <namespace>
kubectl get secret gcp-credentials-dev -n <namespace>
kubectl describe secret gcp-credentials-dev -n <namespace>
```

### Updating the Secret

If you need to update the credentials:

```bash
# Delete the existing secret
kubectl delete secret gcp-credentials -n <namespace>
kubectl delete secret gcp-credentials-dev -n <namespace>

# Recreate it
./create-gcp-credentials-secret.sh prod <namespace>
./create-gcp-credentials-secret.sh dev <namespace>
```

Or update it directly:
```bash
kubectl create secret generic gcp-credentials \
    --from-file=gcp_credential.json \
    --dry-run=client -o yaml | kubectl apply -f - -n <namespace>

kubectl create secret generic gcp-credentials-dev \
    --from-file=gcp_credential_dev.json \
    --dry-run=client -o yaml | kubectl apply -f - -n <namespace>
```

## Deployment

Run the deploy script with the target environment:

```bash
./deploy.sh dev
./deploy.sh prod
```

The deployment automatically mounts the secret in the container:

- Prod: `/app/gcp_credential.json`
- Dev: `/app/gcp_credential_dev.json`

The application will:

1. Prefer the dev credential file in local/dev
2. Fall back to Application Default Credentials (ADC)
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
- Verify the credential file is mounted:
  - Prod: `kubectl exec <pod-name> -n <namespace> -- ls -la /app/gcp_credential.json`
  - Dev: `kubectl exec <pod-name> -n <namespace> -- ls -la /app/gcp_credential_dev.json`
- Verify the credential file is readable without displaying its contents:
  - Prod: `kubectl exec <pod-name> -n <namespace> -- test -r /app/gcp_credential.json`
  - Dev: `kubectl exec <pod-name> -n <namespace> -- test -r /app/gcp_credential_dev.json`
- Inspect file metadata only when troubleshooting permissions:
  - Prod: `kubectl exec <pod-name> -n <namespace> -- stat -c '%a %U:%G %n' /app/gcp_credential.json`
  - Dev: `kubectl exec <pod-name> -n <namespace> -- stat -c '%a %U:%G %n' /app/gcp_credential_dev.json`

Never print, decode, or copy Kubernetes Secret values into terminals, tickets, or CI logs.

## Environment Differences

### Local Development
- Uses `gcp_credential_dev.json`
- Automatically starts Cloud SQL Proxy
- Connects to `127.0.0.1:5432` for database
- Hub URL: `http://localhost:3001` (default)

### Dev (Kubernetes)
- Uses `gcp-credentials-dev` secret (with `dev-` prefix)
- Uses prefixed secret names (e.g., `dev-claude-api-key`)
- Connects via `cloud-sql-proxy-service` Kubernetes service
- Hub URL: `http://leanworks-hub-service` (in-cluster)
- Domain: `dev.leanworks.ai`

### Prod (Kubernetes)
- Uses `gcp-credentials` secret (no prefix)
- Uses base secret names (e.g., `claude-api-key`, not `dev-claude-api-key`)
- Connects via `cloud-sql-proxy-service` Kubernetes service
- Hub URL: `http://leanworks-hub-service` (in-cluster)
- Domain: `leanworks.ai` and `hub.leanworks.ai`
