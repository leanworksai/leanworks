"""
Tests for Document Workflow Orchestrator.

This module tests the advanced document creation and editing workflows including:
- TOC generation and analysis
- Section-by-section drafting with context sandwiches
- Quality passes (continuity, formatting, compression)
- Update strategy detection
- Targeted edits with search and diff
- Broad updates with impact mapping
- RAG fallback for large documents
- Post-update validation
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch
from leanworks.agent.tools.doc_management import DocManagementTool


@pytest.fixture
def mock_tools():
    """Create mock tool instances."""
    postgres_wrapper = Mock()
    postgres_wrapper.org_slug = 'test-org'
    rag_storage = Mock()
    search_tool = Mock()
    bash_tool = Mock()
    text_editor = Mock()
    
    return {
        'postgres_wrapper': postgres_wrapper,
        'rag_storage': rag_storage,
        'search_tool': search_tool,
        'bash_tool': bash_tool,
        'text_editor': text_editor
    }


@pytest.fixture
def orchestrator(mock_tools):
    """Create a DocManagementTool instance with workflow dependencies."""
    # Mock Anthropic client for token counting
    mock_model_client = Mock()
    mock_count_response = Mock()
    mock_count_response.input_tokens = 250  # Mock token count
    mock_model_client.messages.count_tokens.return_value = mock_count_response
    
    return DocManagementTool(
        postgres_client_wrapper=mock_tools['postgres_wrapper'],
        user_id='test-user',
        rag_storage_tool=mock_tools['rag_storage'],
        search_tool=mock_tools['search_tool'],
        bash_tool=mock_tools['bash_tool'],
        text_editor_tool=mock_tools['text_editor'],
        model_client=mock_model_client
    )


class TestTokenEstimation:
    """Test token estimation utilities."""
    
    def test_estimate_tokens_with_api(self, orchestrator):
        """Test token estimation using API."""
        text = "This is a test sentence with approximately twenty characters per word."
        tokens = orchestrator.estimate_tokens(text, use_api=True)
        
        # Should use mocked API response
        assert tokens == 250  # From mock
    
    def test_estimate_tokens_fallback(self, orchestrator):
        """Test token estimation with fallback when API unavailable."""
        text = "This is a test sentence."
        tokens = orchestrator.estimate_tokens(text, use_api=False)
        
        # Should use character-based fallback
        assert tokens == len(text) // 4
        assert tokens > 0
    
    def test_fits_in_context_small(self, orchestrator):
        """Test that small text fits in context."""
        text = "Short text" * 100  # ~1000 chars = ~250 tokens
        assert orchestrator.fits_in_context(text) is True
    
    def test_fits_in_context_large(self, orchestrator):
        """Test that very large text doesn't fit."""
        text = "x" * 200000  # 200K chars = ~50K tokens (exceeds default 30K)
        assert orchestrator.fits_in_context(text) is False
    
    def test_extract_last_n_tokens(self, orchestrator):
        """Test extracting last N tokens from text."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3.\n\nParagraph 4."
        excerpt = orchestrator.extract_last_n_tokens(text, 20)
        
        assert len(excerpt) > 0
        assert len(excerpt) <= 20 * 4  # tokens * avg_chars_per_token
        assert "Paragraph 4" in excerpt
    
    def test_extract_first_n_tokens(self, orchestrator):
        """Test extracting first N tokens from text."""
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3.\n\nParagraph 4."
        excerpt = orchestrator.extract_first_n_tokens(text, 20)
        
        assert len(excerpt) > 0
        assert len(excerpt) <= 20 * 4
        assert "Paragraph 1" in excerpt


class TestTOCGeneration:
    """Test Table of Contents generation."""
    
    def test_generate_toc_clear_structure(self, orchestrator):
        """Test TOC generation with clear requirements."""
        requirements = """
        Create a technical document with the following sections:
        1. Introduction
        2. Architecture Overview
        3. Implementation Details
        4. Testing Strategy
        5. Deployment Guide
        """
        
        result = orchestrator.generate_toc(
            title="System Documentation",
            requirements=requirements
        )
        
        assert result['title'] == "System Documentation"
        assert result['requirements_clarity'] == 'clear'
        assert 'contract' in result
        assert 'sections' in result
        assert result['max_depth'] == 3
    
    def test_generate_toc_exploratory(self, orchestrator):
        """Test TOC generation with vague requirements."""
        requirements = "Write something about machine learning"
        
        result = orchestrator.generate_toc(
            title="ML Overview",
            requirements=requirements
        )
        
        assert result['title'] == "ML Overview"
        assert result['requirements_clarity'] == 'exploratory'
        assert 'Discovery TOC' in result['instructions']
    
    def test_toc_to_markdown(self, orchestrator):
        """Test converting TOC structure to markdown."""
        toc = {
            'title': 'Test Doc',
            'contract': {
                'purpose': 'Testing',
                'audience': 'Developers',
                'scope': 'Unit tests',
                'non_goals': 'Integration tests',
                'evidence_rule': 'Must cite sources'
            },
            'sections': [
                {
                    'heading': 'Section 1',
                    'description': 'First section',
                    'subsections': [
                        {'heading': 'Subsection 1.1', 'description': 'Details'}
                    ]
                }
            ]
        }
        
        markdown = orchestrator._toc_to_markdown(toc)
        
        assert '# Test Doc' in markdown
        assert '## Document Contract' in markdown
        assert 'Testing' in markdown
        assert 'Section 1' in markdown
        assert 'Subsection 1.1' in markdown


class TestSectionDrafting:
    """Test section drafting with context sandwiches."""
    
    def test_prepare_section_context(self, orchestrator):
        """Test preparing context for section drafting."""
        section_info = {
            'heading': 'Implementation Details',
            'description': 'Detailed implementation information',
            'outline': 'Cover architecture, code structure, dependencies'
        }
        previous_content = "Previous section content. " * 100
        next_heading = "Testing Strategy"
        
        result = orchestrator.prepare_section_context(
            section_info=section_info,
            previous_content=previous_content,
            next_section_heading=next_heading
        )
        
        assert result['section_info'] == section_info
        assert result['next_section_heading'] == next_heading
        assert 'context_above' in result
        assert 'drafting_prompt' in result
        assert 'Implementation Details' in result['drafting_prompt']
        assert 'Testing Strategy' in result['drafting_prompt']
    
    def test_get_section_list_from_toc(self, orchestrator):
        """Test extracting flat section list from TOC."""
        toc = {
            'sections': [
                {
                    'heading': 'Section 1',
                    'description': 'First',
                    'subsections': [
                        {'heading': 'Section 1.1', 'description': 'Sub 1'},
                        {'heading': 'Section 1.2', 'description': 'Sub 2'}
                    ]
                },
                {'heading': 'Section 2', 'description': 'Second'}
            ]
        }
        
        sections = orchestrator.get_section_list_from_toc(toc)
        
        assert len(sections) == 4  # 1, 1.1, 1.2, 2
        assert sections[0]['id'] == '1'
        assert sections[1]['id'] == '1.1'
        assert sections[2]['id'] == '1.2'
        assert sections[3]['id'] == '2'
    
    def test_upsert_section_to_file(self, orchestrator):
        """Test upserting section content to file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_file = f.name
            f.write("# Initial Content\n\nFirst section.")
        
        try:
            result = orchestrator.upsert_section_to_file(
                file_path=temp_file,
                section_content="## New Section\n\nNew content here.",
                section_id="2"
            )
            
            assert result['success'] is True
            assert result['section_id'] == '2'
            
            # Verify content was appended
            with open(temp_file, 'r') as f:
                content = f.read()
            
            assert 'Initial Content' in content
            assert 'New Section' in content
            assert 'New content here' in content
        finally:
            os.unlink(temp_file)


class TestQualityPasses:
    """Test quality validation passes."""
    
    def test_run_quality_passes_all(self, orchestrator):
        """Test running all quality passes."""
        content = """# Document Title

## Section 1

Some content with API calls and api references.

Section 2 references Section 3.

This is a test. It should be clear."""
        
        result = orchestrator.run_quality_passes(content)
        
        assert 'passes_run' in result
        assert 'continuity' in result['passes_run']
        assert 'formatting' in result['passes_run']
        assert 'issues_by_pass' in result
    
    def test_continuity_pass_term_variations(self, orchestrator):
        """Test detecting term inconsistencies."""
        content = "The API returns JSON data. The api uses json format. The Api handles JSON."
        
        result = orchestrator._run_continuity_pass(content)
        
        assert len(result['issues']) > 0
        # Should detect API/api/Api variations
    
    def test_formatting_pass_heading_levels(self, orchestrator):
        """Test detecting skipped heading levels."""
        content = """# Title

#### Subsection (skipped H2 and H3)

Some content."""
        
        result = orchestrator._run_formatting_pass(content)
        
        # Should detect heading level skip
        assert any('heading_level_skip' in str(issue) for issue in result['issues'])
    
    def test_compression_pass_large_doc(self, orchestrator):
        """Test compression suggestions for large doc."""
        # Create a large document
        content = "Paragraph content. " * 10000  # ~200K chars
        
        result = orchestrator._run_compression_pass(content)
        
        assert 'suggestions' in result
        assert len(result['suggestions']) > 0


class TestUpdateStrategyDetection:
    """Test update strategy detection."""
    
    def test_detect_direct_update(self, orchestrator):
        """Test detecting direct update for small docs."""
        content = "Short document content"
        update_request = "Add more information"
        
        strategy = orchestrator._detect_update_strategy(content, update_request)
        
        assert strategy['strategy'] == 'direct'
        assert strategy['fits_in_context'] is True
    
    def test_detect_targeted_edit(self, orchestrator):
        """Test detecting targeted edit strategy."""
        content = "x" * 200000  # Large doc
        update_request = 'Update the section about "authentication" to include OAuth'
        
        strategy = orchestrator._detect_update_strategy(content, update_request)
        
        assert strategy['strategy'] == 'targeted'
        assert strategy['has_specific_target'] is True
    
    def test_detect_broad_update(self, orchestrator):
        """Test detecting broad update strategy."""
        content = "x" * 200000  # Large doc
        update_request = "Update document with new API changes"
        
        strategy = orchestrator._detect_update_strategy(content, update_request)
        
        assert strategy['strategy'] == 'broad'
        assert strategy['has_specific_target'] is False


class TestTargetedEditWorkflow:
    """Test targeted edit workflow."""
    
    def test_export_doc_to_temp_file(self, orchestrator):
        """Test exporting doc to temp file."""
        content = "# Document\n\nContent here."
        
        result = orchestrator.export_doc_to_temp_file('test-doc-123', content)
        
        assert result['success'] is True
        assert 'file_path' in result
        assert os.path.exists(result['file_path'])
        
        # Cleanup
        os.unlink(result['file_path'])
    
    def test_search_in_doc_exact_match(self, orchestrator):
        """Test searching for exact text in document."""
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_file = f.name
            f.write("""# Document

## Introduction

This is the introduction section.

## Implementation

Here are the implementation details.

## Conclusion

Final thoughts.""")
        
        try:
            result = orchestrator.search_in_doc(
                file_path=temp_file,
                search_target="implementation details"
            )
            
            assert result['found'] is True
            assert result['match_type'] == 'exact'
            assert 'Implementation' in result['context_window']
        finally:
            os.unlink(temp_file)
    
    def test_search_in_doc_fuzzy_match(self, orchestrator):
        """Test fuzzy search in document."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_file = f.name
            f.write("The quick brown fox jumps over the lazy dog.")
        
        try:
            result = orchestrator.search_in_doc(
                file_path=temp_file,
                search_target="quick fox jumps"  # Not exact, but similar words
            )
            
            # Should find via fuzzy match
            assert result['found'] is True
        finally:
            os.unlink(temp_file)
    
    def test_apply_diff_edit(self, orchestrator):
        """Test applying diff-first edit."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
            temp_file = f.name
            f.write("Original text to replace.")
        
        try:
            result = orchestrator.apply_diff_edit(
                file_path=temp_file,
                old_block="Original text",
                new_block="Updated text"
            )
            
            assert result['success'] is True
            
            # Verify change
            with open(temp_file, 'r') as f:
                content = f.read()
            
            assert "Updated text" in content
            assert "Original text" not in content
        finally:
            os.unlink(temp_file)


class TestRAGFallback:
    """Test RAG fallback for large documents."""
    
    def test_chunk_by_headings(self, orchestrator):
        """Test chunking document by headings."""
        content = """# Main Title

## Section 1

Content for section 1.

### Subsection 1.1

Details for subsection 1.1.

## Section 2

Content for section 2.
"""
        
        chunks = orchestrator._chunk_by_headings(content)
        
        assert len(chunks) > 0
        # Should have chunks for Main Title, Section 1, Subsection 1.1, Section 2
        assert any('Section 1' in chunk.get('heading', '') for chunk in chunks)
        assert any('Section 2' in chunk.get('heading', '') for chunk in chunks)
    
    def test_chunk_by_paragraphs(self, orchestrator):
        """Test chunking document by paragraphs."""
        paragraphs = ["Paragraph {}. ".format(i) * 50 for i in range(10)]
        content = "\n\n".join(paragraphs)
        
        chunks = orchestrator._chunk_by_paragraphs(content)
        
        assert len(chunks) > 0
        # Should create multiple chunks due to token limits
        for chunk in chunks:
            assert chunk['tokens'] <= 600  # Should be around chunk_size + overlap


class TestPostUpdateValidation:
    """Test post-update validation."""
    
    def test_validate_document_update(self, orchestrator):
        """Test validating document after update."""
        original = "The API uses JSON format. Section 2 discusses authentication."
        updated = "The API uses JSON format. Section 2 covers security."
        
        result = orchestrator.validate_document_update(
            doc_id='test-doc',
            original_content=original,
            updated_content=updated,
            change_description="Updated security section"
        )
        
        assert 'doc_id' in result
        assert 'timestamp' in result
        assert 'checks' in result
        assert 'valid' in result
    
    def test_check_for_contradictions(self, orchestrator):
        """Test detecting contradictions."""
        original = "The system is secure. Authentication is required."
        updated = "The system is not secure. Authentication is not required."
        
        contradictions = orchestrator._check_for_contradictions(original, updated)
        
        # Should detect negation flips
        assert len(contradictions) > 0
    
    def test_update_change_log(self, orchestrator):
        """Test generating change log entry."""
        change_entry = {
            'changed_sections': ['Section 2', 'Section 3'],
            'reason': 'Updated API documentation',
            'source': 'API v2.0 spec',
            'issues': 'None detected'
        }
        
        result = orchestrator.update_change_log('test-doc', change_entry)
        
        assert result['success'] is True
        assert 'log_entry' in result
        assert 'Section 2' in result['log_entry']
        assert 'API v2.0 spec' in result['log_entry']


class TestWorkflowEntryPoints:
    """Test main workflow entry points."""
    
    def test_create_doc_with_workflow(self, orchestrator):
        """Test initiating document creation workflow."""
        result = orchestrator.create_doc_with_workflow(
            title="Test Document",
            requirements="Create a technical guide with 3 main sections"
        )
        
        assert result['workflow_initiated'] is True
        assert result['next_step'] == 'generate_toc'
        assert 'instructions' in result
    
    def test_update_doc_with_workflow_small_doc(self, orchestrator, mock_tools):
        """Test initiating update workflow for small doc."""
        mock_tools['doc_tool'].get_doc.return_value = {
            'content': 'Small document content',
            'title': 'Test'
        }
        
        result = orchestrator.update_doc_with_workflow(
            docId='test-doc',
            update_request='Add more details'
        )
        
        assert result['workflow_initiated'] is True
        assert result['strategy'] == 'direct'
        assert result['fits_in_context'] is True
    
    def test_update_doc_with_workflow_large_doc(self, orchestrator, mock_tools):
        """Test initiating update workflow for large doc."""
        large_content = "x" * 200000
        mock_tools['doc_tool'].get_doc.return_value = {
            'content': large_content,
            'title': 'Large Doc'
        }
        
        result = orchestrator.update_doc_with_workflow(
            docId='large-doc',
            update_request='Update the authentication section with OAuth details'
        )
        
        assert result['workflow_initiated'] is True
        assert result['strategy'] == 'targeted'
        assert result['has_specific_target'] is True


class TestCleanup:
    """Test resource cleanup."""
    
    def test_cleanup_temp_files(self, orchestrator):
        """Test cleaning up temporary files."""
        # Create temp files
        temp_file1 = tempfile.NamedTemporaryFile(delete=False)
        temp_file2 = tempfile.NamedTemporaryFile(delete=False)
        
        temp_file1.close()
        temp_file2.close()
        
        orchestrator._temp_files = [temp_file1.name, temp_file2.name]
        
        # Cleanup
        orchestrator.cleanup_temp_files()
        
        assert len(orchestrator._temp_files) == 0
        assert not os.path.exists(temp_file1.name)
        assert not os.path.exists(temp_file2.name)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
