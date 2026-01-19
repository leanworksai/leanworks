"""
Tests for WorkingContext class
"""
import pytest
import time
from datetime import datetime, timedelta
from leanworks.agent.working_context import WorkingContext


class TestWorkingContext:
    """Test cases for WorkingContext functionality"""

    def test_register_and_get_resource(self):
        """Test registering and retrieving resources"""
        wc = WorkingContext()

        # Register a resource
        wc.register_resource(
            resource_id="test_file_1",
            resource_type="temp_file",
            path="/tmp/test.json",
            metadata={"size": 1024}
        )

        # Retrieve the resource
        resource = wc.get_resource("test_file_1")
        assert resource is not None
        assert resource["type"] == "temp_file"
        assert resource["path"] == "/tmp/test.json"
        assert resource["metadata"]["size"] == 1024

    def test_touch_resource(self):
        """Test updating resource last_used timestamp"""
        wc = WorkingContext()

        # Register resource
        wc.register_resource("test_1", "temp_file", "/tmp/test.json")

        # Get initial timestamp
        resource = wc.get_resource("test_1")
        initial_time = resource["last_used"]

        # Wait a bit
        time.sleep(0.01)

        # Touch resource (should update timestamp)
        assert wc.touch_resource("test_1") == True

        # Check timestamp was updated
        resource = wc.get_resource("test_1")
        assert resource["last_used"] > initial_time

        # Touch non-existent resource
        assert wc.touch_resource("nonexistent") == False

    def test_ttl_cleanup(self):
        """Test TTL-based cleanup of expired resources"""
        # Create context with very short TTL for testing
        wc = WorkingContext(default_ttl_hours=0.001)  # ~3.6 seconds

        # Register resources
        wc.register_resource("temp_1", "temp_file", "/tmp/file1.json")
        wc.register_resource("temp_2", "temp_file", "/tmp/file2.json")

        # Should have 2 resources initially
        assert wc.get_resource_count() == 2

        # Wait for expiration
        time.sleep(0.1)  # 100ms > 3.6 seconds TTL

        # Cleanup should remove expired resources
        removed_count = wc.cleanup_expired()
        assert removed_count == 2
        assert wc.get_resource_count() == 0

    def test_get_active_resources(self):
        """Test formatting active resources for context injection"""
        wc = WorkingContext()

        # Register some resources
        wc.register_resource("file1", "temp_file", "/tmp/data.json")
        wc.register_resource("file2", "document_id", "doc-12345")
        wc.register_resource("db1", "storage_ref", "duckdb:response_abc")

        # Get formatted resources
        formatted = wc.get_active_resources()
        assert "Active Working Resources:" in formatted
        assert "/tmp/data.json" in formatted
        assert "doc-12345" in formatted
        assert "duckdb:response_abc" in formatted

    def test_clear(self):
        """Test clearing all resources"""
        wc = WorkingContext()

        # Register resources
        wc.register_resource("res1", "temp_file", "/tmp/test1.json")
        wc.register_resource("res2", "temp_file", "/tmp/test2.json")

        assert wc.get_resource_count() == 2

        # Clear all resources
        wc.clear()
        assert wc.get_resource_count() == 0

        # Verify resources are gone
        assert wc.get_resource("res1") is None
        assert wc.get_resource("res2") is None

    def test_resource_type_ttl_config(self):
        """Test that different resource types have different TTL settings"""
        wc = WorkingContext()

        # Register different types
        wc.register_resource("temp_file", "temp_file", "/tmp/file.json")
        wc.register_resource("doc_id", "document_id", "doc-123")
        wc.register_resource("storage", "storage_ref", "db:ref")

        # Check TTLs were set correctly
        temp_resource = wc.get_resource("temp_file")
        doc_resource = wc.get_resource("doc_id")
        storage_resource = wc.get_resource("storage")

        assert temp_resource["ttl_hours"] == 12  # temp_file TTL
        assert doc_resource["ttl_hours"] == 48   # document_id TTL
        assert storage_resource["ttl_hours"] == 24  # storage_ref TTL

    def test_list_resources(self):
        """Test listing all resources with metadata"""
        wc = WorkingContext()

        wc.register_resource("res1", "temp_file", "/tmp/file.json", {"size": 100})
        wc.register_resource("res2", "document_id", "doc-123")

        resources = wc.list_resources()
        assert len(resources) == 2

        # Check structure
        res1 = next(r for r in resources if r["resource_id"] == "res1")
        assert res1["type"] == "temp_file"
        assert res1["path"] == "/tmp/file.json"
        assert res1["metadata"]["size"] == 100

    def test_max_resources_in_context(self):
        """Test limiting resources shown in context"""
        wc = WorkingContext()

        # Register many resources
        for i in range(25):
            wc.register_resource(f"res{i}", "temp_file", f"/tmp/file{i}.json")

        # Get active resources (should be limited)
        formatted = wc.get_active_resources(max_resources=10)
        lines = formatted.strip().split('\n')

        # Should have header + 10 resources (not all 25)
        assert len(lines) <= 12  # header + 10 resources + possible "..."
        assert any("file0.json" in line for line in lines)
        assert not any("file20.json" in line for line in lines)  # Should not include later ones