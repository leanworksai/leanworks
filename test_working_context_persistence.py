#!/usr/bin/env python3
"""
Test script for working context persistence functionality.
This test verifies that working context is properly saved and loaded across sessions.
"""

import os
import sys
import tempfile
import json
from datetime import datetime, timedelta

# Add the project to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from leanworks.agent.working_context import WorkingContext
from leanworks.utils.cache import firestore_cache

def test_working_context_serialization():
    """Test serialization and deserialization of working context"""
    print("Testing WorkingContext serialization...")
    
    # Create original context with test resources
    original = WorkingContext()
    
    # Add some test resources
    original.register_resource("test_file_1", "temp_file", "/tmp/test1.txt", {"description": "Test file 1"})
    original.register_resource("test_doc_1", "document_id", "doc-abc123", {"title": "Test Document"})
    original.register_resource("test_storage_1", "storage_ref", "bucket/path/file.pdf", {"size": 1024})
    
    # Serialize to dict
    state_dict = original.to_dict()
    
    # Verify serialization
    assert "resources" in state_dict
    assert "default_ttl_hours" in state_dict
    assert "ttl_config" in state_dict
    assert len(state_dict["resources"]) == 3
    
    # Deserialize from dict
    restored = WorkingContext.from_dict(state_dict)
    
    # Verify deserialization
    assert restored.get_resource_count() == 3
    assert "test_file_1" in restored.resources
    assert "test_doc_1" in restored.resources
    assert "test_storage_1" in restored.resources
    
    print("✅ WorkingContext serialization test passed")

def test_working_context_validation():
    """Test resource validation functionality"""
    print("Testing WorkingContext resource validation...")
    
    # Create context with some valid and invalid resources
    wc = WorkingContext()
    
    # Create a real temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(b"test content")
        tmp_path = tmp_file.name
    
    # Add valid resource
    wc.register_resource("valid_file", "temp_file", tmp_path, {"valid": True})
    
    # Add invalid resource (non-existent file)
    wc.register_resource("invalid_file", "temp_file", "/tmp/nonexistent.txt", {"invalid": True})
    
    # Validate resources
    removed_count = wc.validate_resources()
    
    # Should remove 1 invalid resource
    assert removed_count == 1
    assert wc.get_resource_count() == 1
    assert "valid_file" in wc.resources
    assert "invalid_file" not in wc.resources
    
    # Cleanup
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)
    
    print("✅ WorkingContext validation test passed")

def test_working_context_expiration():
    """Test resource expiration cleanup"""
    print("Testing WorkingContext expiration cleanup...")
    
    wc = WorkingContext()
    
    # Create a resource that expires soon
    wc.register_resource("expired_resource", "temp_file", "/tmp/expired.txt", {"old": True})
    
    # Manually set the last_used time to be expired
    expired_time = datetime.now() - timedelta(hours=13)  # Past the 12h TTL
    wc.resources["expired_resource"]["last_used"] = expired_time
    
    # Add another resource that's still valid
    wc.register_resource("valid_resource", "document_id", "doc-valid", {"new": True})
    
    # Clean up expired resources
    removed_count = wc.cleanup_expired()
    
    # Should remove 1 expired resource
    assert removed_count == 1
    assert wc.get_resource_count() == 1
    assert "expired_resource" not in wc.resources
    assert "valid_resource" in wc.resources
    
    print("✅ WorkingContext expiration test passed")

def main():
    """Run all tests"""
    print("Starting WorkingContext persistence tests...\n")
    
    try:
        test_working_context_serialization()
        test_working_context_validation()
        test_working_context_expiration()
        
        print("\n🎉 All tests passed! WorkingContext persistence is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)