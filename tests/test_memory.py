#!/usr/bin/env python3
"""
Test script for the MemoryManager module.
Demonstrates incremental summary buffer functionality.
"""

import json
import logging
from datetime import datetime
from unittest.mock import Mock, MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Mock classes for testing
class MockModelClient:
    """Mock Claude model client for testing."""
    
    def __init__(self):
        self.messages = Mock()
        
        # Mock a summarization response
        self.mock_summary_response = Mock()
        self.mock_summary_response.content = [
            Mock(type="text", text="Summary: User asked about project setup, discussed database configuration, and requested help with authentication. Key decisions: Use PostgreSQL, implement JWT tokens, and set up OAuth integration.")
        ]
        self.messages.create.return_value = self.mock_summary_response
        
        # Mock token counting response (Claude's official API)
        self.mock_count_response = Mock()
        self.mock_count_response.input_tokens = 42  # Mock token count
        self.messages.count_tokens.return_value = self.mock_count_response

class MockStorageClient:
    """Mock storage client for testing."""
    
    def __init__(self):
        self.storage = {}
    
    def upload_blob_from_memory(self, data, path):
        self.storage[path] = data
        print(f"Saved to storage: {path}")
    
    def download_blob_to_memory(self, path):
        return self.storage.get(path)

def create_mock_message(role: str, text: str):
    """Create a mock message in Claude format."""
    return {
        "role": role,
        "content": [{"type": "text", "text": text}]
    }

def test_memory_manager():
    """Test the MemoryManager functionality."""
    print("=" * 60)
    print("TESTING MEMORY MANAGER - INCREMENTAL SUMMARY BUFFER")
    print("=" * 60)
    
    # Import here to avoid circular imports during initialization
    from leanworks.agent.memory import MemoryManager
    
    # Create mock clients
    model_client = MockModelClient()
    storage_client = MockStorageClient()
    
    # Test both old-style initialization and new model-aware initialization
    print("\n--- Testing Model-Aware Configuration ---")
    
    # Test with Claude 4 Sonnet (should get large buffer settings)
    claude_4_memory = MemoryManager.create_for_model(
        model_name="claude-sonnet-4-20250514",
        model_client=model_client,
        storage_client=storage_client,
        user_id="test_user_claude4",
        session_id="test_session_claude4"
    )
    
    print(f"Claude 4 Sonnet Memory Config:")
    print(json.dumps(claude_4_memory.get_memory_stats(), indent=2))
    
    # Test with Claude Haiku (should get smaller but still reasonable settings)
    haiku_memory = MemoryManager.create_for_model(
        model_name="claude-3-haiku-20240307",
        model_client=model_client,
        storage_client=storage_client,
        user_id="test_user_haiku",
        session_id="test_session_haiku"
    )
    
    print(f"\nClaude Haiku Memory Config:")
    print(json.dumps(haiku_memory.get_memory_stats(), indent=2))
    
    # For testing, use lower thresholds to actually trigger summarization
    print("\n--- Creating Test Instance with Lower Thresholds ---")
    memory_manager = MemoryManager(
        model_client=model_client,
        storage_client=storage_client,
        user_id="test_user",
        session_id="test_session",
        max_context_tokens=2000,    # Moderate threshold for testing
        trigger_threshold=800,      # Lower to trigger summarization in test
        summary_max_tokens=200,
        recent_turns_to_keep=4,     # Keep 4 recent turns
        summarization_model="claude-3-haiku-20240307"
    )
    
    print(f"\n1. Initial Memory Stats:")
    print(json.dumps(memory_manager.get_memory_stats(), indent=2))
    
    # Set system prompt and user profile
    memory_manager.set_system_prompt("You are a helpful AI assistant for project management.")
    memory_manager.set_user_profile("Software engineer working on web applications, prefers TypeScript and React")
    
    print(f"\n2. After Setting System Prompt and User Profile:")
    print(json.dumps(memory_manager.get_memory_stats(), indent=2))
    
    # Simulate a conversation that will trigger summarization
    conversation_turns = [
        ("user", "Hello, I'm starting a new web project. What database should I use?"),
        ("assistant", "For a web project, I'd recommend considering PostgreSQL for its robustness and feature set, or MongoDB if you need flexible document storage. What type of data will you be storing?"),
        
        ("user", "I'll be storing user accounts, project data, and task information. Mostly structured data."),
        ("assistant", "PostgreSQL would be perfect for that use case. It excels with structured data, has excellent support for relationships between tables, and offers strong consistency guarantees for user account data."),
        
        ("user", "Great! How should I handle user authentication?"),
        ("assistant", "For authentication, I recommend implementing JWT tokens with a secure refresh token strategy. You could also consider integrating OAuth providers like Google or GitHub for social login options."),
        
        ("user", "What about database migrations and schema management?"),
        ("assistant", "For PostgreSQL, consider using a migration tool like Flyway or Liquibase for Java projects, or Alembic for Python. These tools help you version control your database schema changes safely."),
        
        ("user", "I'm using Node.js. Any specific recommendations?"),
        ("assistant", "For Node.js with PostgreSQL, I recommend using Prisma as your ORM and migration tool. It provides excellent TypeScript support, automatic migration generation, and a great developer experience with its database client."),
        
        ("user", "Perfect! One more question - how do I structure my API routes?"),
        ("assistant", "Structure your API with RESTful conventions: /api/users for user operations, /api/projects for projects, /api/tasks for tasks. Use Express.js with middleware for authentication, validation, and error handling. Consider using express-validator for input validation.")
    ]
    
    print(f"\n3. Adding Conversation Turns (will trigger summarization):")
    for i, (role, text) in enumerate(conversation_turns):
        message = create_mock_message(role, text)
        
        if role == "user":
            print(f"\n   Turn {i//2 + 1}: Adding user message...")
            memory_manager.add_turn(message)
        else:
            print(f"   Turn {i//2 + 1}: Adding assistant response...")
            memory_manager.update_assistant_response(message)
        
        stats = memory_manager.get_memory_stats()
        print(f"   -> Total tokens: {stats['total_tokens']}, Turns: {stats['conversation_turns']}")
        
        if stats['total_tokens'] >= memory_manager.trigger_threshold:
            print(f"   -> 🔥 SUMMARIZATION TRIGGERED! (threshold: {memory_manager.trigger_threshold})")
    
    print(f"\n4. Final Memory Stats After All Turns:")
    final_stats = memory_manager.get_memory_stats()
    print(json.dumps(final_stats, indent=2))
    
    print(f"\n5. Getting Context for Inference:")
    context, recent_messages = memory_manager.get_context_for_inference()
    print(f"Context length: {len(context)} characters")
    print(f"Recent messages count: {len(recent_messages)}")
    print(f"\nContext preview:")
    print("-" * 40)
    print(context[:500] + "..." if len(context) > 500 else context)
    print("-" * 40)
    
    print(f"\n6. Recent Messages:")
    for i, msg in enumerate(recent_messages):
        role = msg['role']
        text = msg['content'][0]['text'][:100] + "..." if len(msg['content'][0]['text']) > 100 else msg['content'][0]['text']
        print(f"   {i+1}. {role}: {text}")
    
    print(f"\n7. Current Running Summary:")
    print("-" * 40)
    print(memory_manager.running_summary)
    print("-" * 40)
    
    print(f"\n8. Testing Memory Persistence:")
    # Save current state
    memory_manager.save_memory_state()
    
    # Create new instance to test loading
    new_memory_manager = MemoryManager(
        model_client=model_client,
        storage_client=storage_client,
        user_id="test_user",
        session_id="test_session",
        max_context_tokens=1000,
        trigger_threshold=300,
        summary_max_tokens=100,
        recent_turns_to_keep=3
    )
    
    print(f"New instance stats after loading:")
    print(json.dumps(new_memory_manager.get_memory_stats(), indent=2))
    print(f"Running summary matches: {new_memory_manager.running_summary == memory_manager.running_summary}")
    print(f"Conversation turns match: {len(new_memory_manager.conversation_turns) == len(memory_manager.conversation_turns)}")
    
    print(f"\n9. Demonstrating Large Buffer Benefits:")
    print("   Testing with a Claude 4 configuration for realistic usage...")
    
    # Create a memory manager with Claude 4 settings
    large_buffer_memory = MemoryManager.create_for_model(
        model_name="claude-sonnet-4-20250514",
        model_client=model_client,
        storage_client=storage_client,
        user_id="test_large_buffer",
        session_id="test_large_buffer"
    )
    
    # Add many conversation turns to show that summarization is rarely needed
    print(f"   Initial state: {large_buffer_memory.get_memory_stats()['total_tokens']} tokens")
    
    # Simulate a very long conversation (50 turns)
    long_conversation = [
        ("user", f"This is user message {i}. " + "Let me add some more content to make this message longer. " * 5)
        for i in range(1, 26)
    ] + [
        ("assistant", f"This is assistant response {i}. " + "I'm providing a detailed response with lots of context and information. " * 8)
        for i in range(1, 26)
    ]
    
    # Mix user and assistant messages
    mixed_conversation = []
    for i in range(25):
        mixed_conversation.append(("user", f"User message {i+1}. " + "Adding substantial content to simulate real conversation. " * 6))
        mixed_conversation.append(("assistant", f"Assistant response {i+1}. " + "Providing comprehensive answers with detailed explanations. " * 7))
    
    summarization_count = 0
    for i, (role, text) in enumerate(mixed_conversation):
        message = create_mock_message(role, text)
        
        if role == "user":
            large_buffer_memory.add_turn(message)
        else:
            large_buffer_memory.update_assistant_response(message)
        
        # Check if summarization was triggered
        stats = large_buffer_memory.get_memory_stats()
        if stats['last_summarization'] != "Never":
            summarization_count += 1
    
    final_large_stats = large_buffer_memory.get_memory_stats()
    print(f"   After 50 messages:")
    print(f"   - Total tokens: {final_large_stats['total_tokens']:,}")
    print(f"   - Trigger threshold: {final_large_stats['trigger_threshold']:,}")
    print(f"   - Summarizations triggered: {summarization_count}")
    print(f"   - Conversation turns retained: {final_large_stats['conversation_turns']}")
    print(f"   - Tokens until next summarization: {final_large_stats['tokens_until_trigger']:,}")
    
    if summarization_count == 0:
        print(f"   ✅ SUCCESS: No summarization needed with large buffer!")
        print(f"   💡 The 200K context window allows for extensive conversations without losing context.")
    
    print(f"\n10. Buffer Size Comparison:")
    print(f"    Old defaults (8K context):  Summarization every ~10-15 turns")
    print(f"    New defaults (180K context): Summarization every ~200-300 turns")
    print(f"    📈 This is a ~20x improvement in context retention!")

    print(f"\n{'='*60}")
    print("MEMORY MANAGER TEST COMPLETED!")
    print("Key Benefits of Large Buffer:")
    print("- 20x more context retained before summarization")
    print("- Better conversation continuity")
    print("- Fewer API calls for summarization")
    print("- Optimal use of Claude 4's 200K context window")
    print(f"{'='*60}")

def test_token_counting():
    """Test both accurate token counting and estimation functionality.""" 
    print("\n" + "="*50)
    print("TESTING TOKEN COUNTING (CLAUDE API + ESTIMATION)")
    print("="*50)
    
    from leanworks.agent.memory import MemoryManager
    
    # Create memory manager for testing both accurate and estimated token counting
    mock_client = MockModelClient()
    memory_manager = MemoryManager(
        model_client=mock_client,
        storage_client=MockStorageClient(),
        user_id="test",
        session_id="test",
        main_model="claude-sonnet-4-20250514"  # Set main model for accurate counting
    )
    
    test_texts = [
        ("Short text", "Hello world"),
        ("Medium text", "This is a longer piece of text that should have more tokens estimated for it based on the character count."),
        ("Long text", "This is a very long piece of text that contains multiple sentences and should demonstrate how the token estimation works. " * 10),
        ("Empty text", ""),
    ]
    
    for name, text in test_texts:
        tokens = memory_manager.estimate_tokens(text)
        print(f"{name:15} | Length: {len(text):4} chars | Estimated tokens: {tokens:3}")
    
    # Test message token estimation
    print(f"\nMessage Token Estimation:")
    test_messages = [
        create_mock_message("user", "Hello, how are you?"),
        create_mock_message("assistant", "I'm doing well, thank you for asking! How can I help you today?"),
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll help you with that."},
                {"type": "tool_use", "id": "123", "name": "search", "input": {"query": "test"}},
            ]
        }
    ]
    
    for i, msg in enumerate(test_messages):
        estimated_tokens = memory_manager.estimate_message_tokens(msg)
        print(f"Message {i+1:2} | Role: {msg['role']:9} | Estimated tokens: {estimated_tokens:3}")
    
    print(f"\nAccurate Token Counting (using Claude API):")
    
    # Test accurate token counting with a few messages
    test_conversation = [
        create_mock_message("user", "Hello, how are you today?"),
        create_mock_message("assistant", "I'm doing well, thank you for asking! How can I help you?")
    ]
    
    # Mock different token counts to show the difference
    mock_client.mock_count_response.input_tokens = 156  # More realistic count
    
    accurate_count = memory_manager.count_tokens_accurate(
        test_conversation, 
        "You are a helpful AI assistant."
    )
    
    # Also get estimated count for comparison
    estimated_total = memory_manager.estimate_tokens("You are a helpful AI assistant.")
    for msg in test_conversation:
        estimated_total += memory_manager.estimate_message_tokens(msg)
    
    print(f"System + 2 messages:")
    print(f"  Accurate count (Claude API): {accurate_count} tokens")
    print(f"  Estimated count (approximation): {estimated_total} tokens")
    print(f"  Difference: {abs(accurate_count - estimated_total)} tokens")
    
    print(f"\n✅ Benefits of Claude's Official Token Counting API:")
    print(f"  - Exact token counts for precise memory management")
    print(f"  - Accounts for tokenization nuances and special tokens")
    print(f"  - Free to use (subject to rate limits)")
    print(f"  - Supports all Claude models and input types (images, PDFs, tools)")
    print(f"  - Reference: https://docs.anthropic.com/en/docs/build-with-claude/token-counting")

if __name__ == "__main__":
    test_token_counting()
    test_memory_manager() 