-- Migration: Add file_metadata table for tracking files uploaded to Claude Files API
-- Created: 2025-01-08
-- Description: Stores metadata about files uploaded to Claude Files API for audit and tracking

CREATE TABLE IF NOT EXISTS file_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_slug VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    file_id VARCHAR(255) NOT NULL UNIQUE,  -- Claude file_id
    filename VARCHAR(500) NOT NULL,
    mime_type VARCHAR(100) NOT NULL,
    size_bytes BIGINT NOT NULL,
    uploaded_at TIMESTAMP DEFAULT NOW(),
    deleted_at TIMESTAMP,
    metadata JSONB,
    CONSTRAINT file_metadata_file_id_unique UNIQUE (file_id)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_file_metadata_file_id ON file_metadata(file_id);
CREATE INDEX IF NOT EXISTS idx_file_metadata_org_user ON file_metadata(org_slug, user_id);
CREATE INDEX IF NOT EXISTS idx_file_metadata_session ON file_metadata(session_id);
CREATE INDEX IF NOT EXISTS idx_file_metadata_uploaded_at ON file_metadata(uploaded_at);

-- Add comment to table
COMMENT ON TABLE file_metadata IS 'Stores metadata about files uploaded to Claude Files API for audit and tracking purposes';
