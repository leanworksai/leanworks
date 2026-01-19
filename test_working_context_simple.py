#!/usr/bin/env python3
"""
Simple test script for working context persistence functionality.
Tests the core functionality without external dependencies.
"""

import os
import sys
import tempfile
import json
from datetime import datetime, timedelta

# Add the project to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock datetime.now for consistent testing
import datetime as dt_util
monkeypatch = lambda: print("Monkeypatch datetime")

def test_working_context_serialization():
    """Test serialization and deserialization of working context"""
    print("Testing WorkingContext serialization...")
    
    # Import working context
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'leanworks', 'agent'))
    from working_context import WorkingContext
    
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
    
    # Verify metadata preservation
    assert restored.resources["test_file_1"]["path"] == "/tmp/test1.txt"
    assert restored.resources["test_doc_1"]["metadata"]["title"] == "Test Document"

def test_working_context_validation():
    """Test resource validation functionality"""
    print("Testing WorkingContext resource validation...")
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'leanworks', 'agent'))
    from working_context import WorkingContext
    
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

def test_integration_path():
    """Test the complete save/load cycle simulation"""
    print("Testing integration save/load cycle...")
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'leanworks', 'agent'))
    from working_context import WorkingContext
    
    # Create working context with resources
    original = WorkingContext()
    
    # Add test resources
    with tempfile.NamedTemporaryFile(delete=False) as f1:
        f1.write(b"file 1 content")
        path1 = f1.name
    
    with tempfile.NamedTemporaryFile(delete=False) as f2:
        f2.write(b"file 2 content")
        path2 = f2.name
    
    original.register_resource("temp1", "temp_file", path1, {"test": 1})
    original.register_resource("temp2", "temp_file", path2, {"test": 2})
    original.register_resource("doc1", "document_id", "doc-test123", {"title": "Test Doc"})
    
    # Simulate "saving" to Firestore (here we just serialize)
    serialized_state = original.to_dict()
    
    # Simulate starting a new session and "loading" from Firestore
    restored = WorkingContext.from_dict(serialized_state)
    
    # Simulate validation on restart
    removed = restored.validate_resources()
    
    # Cleanup the temp files before validation check
    if os.path.exists(path1):
        os.unlink(path1)
    if os.path.exists(path2):
        os.unlink(path2)
    
    # Now test validation after files are gone
    removed2 = restored.validate_resources()
    
    # Should have 3 resources initially, then 0 after files are removed
    print(f"Initial resources: {original.get_resource_count()}")
    print(f"Restored resources: {restored.get_resource_count()}")
    print(f"Resources after validation: {removed} + {removed2} removed")
    
    assert restored.get_resource_count() == 0

def main():
    """Run all tests"""
    print("Starting WorkingContext persistence tests...\n")
    
    try:
        test_working_context_serialization()
        print("✅ Serialization test passed")
        
        test_working_context_validation()
        print("✅ Validation test passed")
        
        test_integration_path()
        print("✅ Integration test passed")
        
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