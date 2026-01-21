"""
Validation script to verify instruction hierarchy is properly implemented.
Tests that tool syntax appears only in system prompt, not duplicated in tool instructions.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from leanworks.setting import AGENT_SYSTEM_PROMPT, DOC_SIZE_CONFIG
from leanworks.agent.tools.doc_management import DocManagementTool
from leanworks.agent.tools.user_management import UserManagementTool


def test_function_name_consistency():
    """Verify all references use search_tool_response_in_vectorstore"""
    print("Testing function name consistency...")
    
    # Check system prompt
    assert "search_tool_response_in_vectordb" not in AGENT_SYSTEM_PROMPT, \
        "FAIL: Old function name 'vectordb' found in system prompt"
    assert "search_tool_response_in_vectorstore" in AGENT_SYSTEM_PROMPT, \
        "FAIL: Correct function name 'vectorstore' not found in system prompt"
    
    print("✅ Function name consistency: PASS")


def test_system_prompt_has_tool_syntax():
    """Verify system prompt contains detailed tool syntax"""
    print("\nTesting system prompt contains tool syntax...")
    
    # Should contain grep usage
    assert "grep -n 'pattern'" in AGENT_SYSTEM_PROMPT, \
        "FAIL: grep syntax not in system prompt"
    
    # Should contain text_editor usage
    assert "text_editor(path=" in AGENT_SYSTEM_PROMPT, \
        "FAIL: text_editor syntax not in system prompt"
    
    # Should contain jq usage
    assert "jq '.path.to.field'" in AGENT_SYSTEM_PROMPT, \
        "FAIL: jq syntax not in system prompt"
    
    print("✅ System prompt has tool syntax: PASS")


def test_tool_instructions_reference_system():
    """Verify tool instructions reference system prompt instead of repeating syntax"""
    print("\nTesting tool instructions reference system prompt...")
    
    # Create dummy instances
    class DummyWrapper:
        org_slug = "test_org"
    
    doc_tool = DocManagementTool(DummyWrapper(), user_id="test@example.com")
    
    # Get instructions
    create_inst = doc_tool.get_create_doc_instruction()
    update_inst = doc_tool.get_update_doc_instruction()
    understand_inst = doc_tool.get_understand_doc_instruction()
    
    # Should reference system prompt
    assert "<core_tools_reference>" in create_inst, \
        "FAIL: get_create_doc_instruction doesn't reference <core_tools_reference>"
    assert "<core_tools_reference>" in update_inst, \
        "FAIL: get_update_doc_instruction doesn't reference <core_tools_reference>"
    assert "<core_tools_reference>" in understand_inst, \
        "FAIL: get_understand_doc_instruction doesn't reference <core_tools_reference>"
    
    # Should NOT contain detailed syntax
    assert "grep -n 'pattern'" not in update_inst, \
        "FAIL: get_update_doc_instruction contains detailed grep syntax"
    assert "grep -n \"exact text\"" not in update_inst, \
        "FAIL: get_update_doc_instruction contains detailed grep syntax"
    assert "text_editor(command=" not in update_inst, \
        "FAIL: get_update_doc_instruction contains detailed text_editor syntax"
    
    print("✅ Tool instructions reference system: PASS")


def test_no_redundant_prerequisites():
    """Verify prerequisites appear in tool definitions, not duplicated in system"""
    print("\nTesting no redundant prerequisites...")
    
    # System prompt should have high-level guidance only
    assert "call the appropriate instruction tool first" in AGENT_SYSTEM_PROMPT.lower(), \
        "FAIL: High-level guidance missing from system prompt"
    
    # Should NOT have detailed step-by-step in system prompt
    assert "Call get_understand_doc_instruction() FIRST, then get_doc()" not in AGENT_SYSTEM_PROMPT, \
        "FAIL: Detailed prerequisites still in system prompt (should be high-level only)"
    
    print("✅ No redundant prerequisites: PASS")


def test_constants_centralized():
    """Verify constants are centralized in config"""
    print("\nTesting constants centralization...")
    
    # DOC_SIZE_CONFIG should exist
    assert DOC_SIZE_CONFIG is not None, \
        "FAIL: DOC_SIZE_CONFIG not defined"
    assert "small_doc_threshold" in DOC_SIZE_CONFIG, \
        "FAIL: small_doc_threshold not in DOC_SIZE_CONFIG"
    assert DOC_SIZE_CONFIG["small_doc_threshold"] == 8000, \
        "FAIL: small_doc_threshold should be 8000"
    
    print("✅ Constants centralized: PASS")


def test_instruction_hierarchy():
    """Verify clear separation between layers"""
    print("\nTesting instruction hierarchy...")
    
    # Layer 1 (system prompt) should have reference sections
    assert "<workspace_reference>" in AGENT_SYSTEM_PROMPT, \
        "FAIL: <workspace_reference> not in system prompt"
    assert "<core_tools_reference>" in AGENT_SYSTEM_PROMPT, \
        "FAIL: <core_tools_reference> not in system prompt"
    
    # Layer 3 (tool instructions) should reference Layer 1
    class DummyWrapper:
        org_slug = "test_org"
    doc_tool = DocManagementTool(DummyWrapper(), user_id="test@example.com")
    update_inst = doc_tool.get_update_doc_instruction()
    
    # Should have workflow logic
    assert "WORKFLOW A:" in update_inst or "WORKFLOW B:" in update_inst, \
        "FAIL: Tool instructions missing workflow logic"
    
    # Should reference system prompt
    assert "see <core_tools_reference>" in update_inst, \
        "FAIL: Tool instructions don't reference system prompt"
    
    print("✅ Instruction hierarchy: PASS")


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("VALIDATING INSTRUCTION HIERARCHY OPTIMIZATION")
    print("=" * 60)
    
    try:
        test_function_name_consistency()
        test_system_prompt_has_tool_syntax()
        test_tool_instructions_reference_system()
        test_no_redundant_prerequisites()
        test_constants_centralized()
        test_instruction_hierarchy()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✅")
        print("=" * 60)
        print("\nInstruction hierarchy is properly optimized:")
        print("- No function name conflicts")
        print("- No tool syntax redundancy")
        print("- Clear layer separation")
        print("- Constants centralized")
        print("- Maintainable and scalable")
        
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
