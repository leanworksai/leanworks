"""
Integration tests for memory fact preservation during summarization
"""
import pytest
from unittest.mock import Mock, patch
from leanworks.agent.memory import MemoryManager
from leanworks.agent.working_context import WorkingContext
from leanworks.agent.fact_extractor import FactExtractor


class TestMemoryFactPreservation:
    """Integration tests for fact preservation in memory management"""

    @pytest.fixture
    def mock_model_client(self):
        """Mock Claude model client"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Summary preserving key facts: [FACT: analysis results | /tmp/data.json]"
        mock_response.content[0].type = "text"
        mock_client.messages.create.return_value = mock_response
        return mock_client

    @pytest.fixture
    def memory_manager(self, mock_model_client):
        """Create MemoryManager with mocked dependencies"""
        with patch('leanworks.agent.memory.firestore') as mock_firestore:
            manager = MemoryManager(
                model_client=mock_model_client,
                firestore_client=mock_firestore,
                org_slug="test-org",
                user_id="test-user",
                session_id="test-session",
                max_context_tokens=1000,  # Small for testing
                trigger_threshold=500,
                summary_max_tokens=100
            )
            return manager

    def test_fact_extraction_and_registration(self, memory_manager):
        """Test that facts are extracted from turns and registered in working context"""
        # Add conversation turns with extractable facts
        user_message = {
            "role": "user",
            "content": [{"type": "text", "text": "Analyze the data in /tmp/analysis.json"}]
        }
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": "I found results in response_id: abc123def"}]
        }

        memory_manager.add_turn(user_message, assistant_message)

        # Check that facts were extracted and registered
        assert memory_manager.working_context.get_resource_count() > 0

        # Check specific resources
        file_resource = None
        storage_resource = None
        for resource in memory_manager.working_context.list_resources():
            if resource["type"] == "file_paths":
                file_resource = resource
            elif resource["type"] == "storage_refs":
                storage_resource = resource

        assert file_resource is not None
        assert "/tmp/analysis.json" in file_resource["path"]
        assert storage_resource is not None
        assert "response_id:abc123def" in storage_resource["path"]

    def test_summarization_preserves_facts_in_context(self, memory_manager):
        """Test that summarization includes extracted facts in context"""
        # Add enough turns to trigger summarization
        for i in range(25):  # More than recent_turns_to_keep (20)
            user_msg = {
                "role": "user",
                "content": [{"type": "text", "text": f"Query {i} about /tmp/file{i}.json"}]
            }
            assistant_msg = {
                "role": "assistant",
                "content": [{"type": "text", "text": f"Result {i} saved as response_id: id{i}"}]
            }
            memory_manager.add_turn(user_msg, assistant_msg)

        # Trigger summarization by checking token count
        current_tokens = memory_manager._calculate_current_tokens()
        if current_tokens > memory_manager.trigger_threshold:
            memory_manager._perform_summarization()

        # Check that working context still has resources
        assert memory_manager.working_context.get_resource_count() > 0

        # Check that context injection includes working resources
        context_text, messages = memory_manager.get_context_for_inference()
        assert "Active Working Resources:" in context_text

    def test_context_injection_includes_working_resources(self, memory_manager):
        """Test that get_context_for_inference includes working context resources"""
        # Add some resources to working context
        memory_manager.working_context.register_resource(
            "test_file", "temp_file", "/tmp/test.json", {"source": "test"}
        )
        memory_manager.working_context.register_resource(
            "test_doc", "document_id", "doc-12345", {"source": "test"}
        )

        # Get context for inference
        context_text, messages = memory_manager.get_context_for_inference()

        # Check that working resources are included
        assert "Active Working Resources:" in context_text
        assert "/tmp/test.json" in context_text
        assert "doc-12345" in context_text

    def test_resource_cleanup_on_session_end(self, memory_manager):
        """Test that resources are cleaned up when session ends"""
        # Add resources
        memory_manager.working_context.register_resource("res1", "temp_file", "/tmp/test.json")
        memory_manager.working_context.register_resource("res2", "temp_file", "/tmp/test2.json")

        assert memory_manager.working_context.get_resource_count() == 2

        # Simulate session cleanup (like in ChatAgent.__del__)
        memory_manager.working_context.clear()

        assert memory_manager.working_context.get_resource_count() == 0

    def test_ttl_based_cleanup_integration(self, memory_manager):
        """Test TTL cleanup works with memory manager integration"""
        # Override TTL for fast testing
        memory_manager.working_context.ttl_config["temp_file"] = 0.001  # Very short TTL

        # Add resource
        memory_manager.working_context.register_resource("temp", "temp_file", "/tmp/test.json")

        assert memory_manager.working_context.get_resource_count() == 1

        # Wait for expiration
        import time
        time.sleep(0.1)

        # Cleanup should remove expired resource
        removed = memory_manager.working_context.cleanup_expired()
        assert removed == 1
        assert memory_manager.working_context.get_resource_count() == 0

    def test_enhanced_summarization_prompt_includes_facts(self, memory_manager, mock_model_client):
        """Test that enhanced summarization prompt includes extracted facts"""
        # Add turns with facts
        turns_to_summarize = [
            {
                "user_message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Process /tmp/data.json"}]
                },
                "assistant_message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Saved as response_id: abc123"}]
                }
            }
        ]

        # Create enhanced prompt
        facts = FactExtractor.extract_facts(turns_to_summarize)
        prompt = memory_manager._create_enhanced_summarization_prompt("test conversation", facts)

        # Check that prompt includes fact preservation rules
        assert "CRITICAL PRESERVATION RULES:" in prompt
        assert "Extracted Technical Facts:" in prompt
        assert "/tmp/data.json" in prompt or any("/tmp/data.json" in str(fact) for fact in facts.values())

    def test_fact_extraction_edge_cases(self):
        """Test fact extraction handles various edge cases"""
        # Test with malformed messages
        conversation_turns = [
            {"user_message": {"role": "user", "content": None}, "assistant_message": None},
            {"user_message": {"role": "user", "content": "plain string"}, "assistant_message": None},
            {"user_message": {"role": "user", "content": []}, "assistant_message": None}
        ]

        # Should not crash
        facts = FactExtractor.extract_facts(conversation_turns)
        assert isinstance(facts, dict)
        assert all(isinstance(v, list) for v in facts.values())

    def test_working_context_limits_context_injection(self, memory_manager):
        """Test that working context limits resources shown in context"""
        # Add many resources
        for i in range(25):
            memory_manager.working_context.register_resource(
                f"res{i}", "temp_file", f"/tmp/file{i}.json"
            )

        # Get context with limit
        context_text, _ = memory_manager.get_context_for_inference()

        # Should not include all 25 resources (implementation limits to reasonable number)
        lines = context_text.split('\n')
        resource_lines = [line for line in lines if '/tmp/file' in line]
        assert len(resource_lines) <= 20  # Should be limited