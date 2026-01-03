"""
Test script for Notion tool integration.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from leanworks.agent.tools.notion import NotionTool
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_notion_tool():
    """Test Notion tool with provided integration token."""
    # Integration token provided by user
    integration_token = "ntn_I21581120359BCFtaNldlbz8F3N0FHIdbPgGrmUqKYu0NB"
    
    # Initialize NotionTool
    notion_tool = NotionTool(integration_token=integration_token)
    
    print("=" * 80)
    print("Testing Notion Tool")
    print("=" * 80)
    
    # Test 1: Search pages
    print("\n1. Testing search_pages...")
    try:
        search_results = notion_tool.search_pages(query="", page_size=5)
        if 'error' in search_results:
            print(f"   ERROR: {search_results['error']}")
        else:
            print(f"   SUCCESS: Found {len(search_results)} pages")
            if search_results:
                print(f"   First page: {search_results[0].get('title', 'No title')} (ID: {search_results[0].get('id', 'N/A')})")
    except Exception as e:
        print(f"   EXCEPTION: {str(e)}")
    
    # Test 2: Get a page (if we found one)
    print("\n2. Testing get_page...")
    try:
        # First, search for a page to get an ID
        search_results = notion_tool.search_pages(page_size=1)
        if 'error' not in search_results and search_results:
            page_id = search_results[0].get('id')
            if page_id:
                page_result = notion_tool.get_page(page_id)
                if 'error' in page_result:
                    print(f"   ERROR: {page_result['error']}")
                else:
                    print(f"   SUCCESS: Retrieved page {page_id}")
                    print(f"   Title: {page_result.get('title', 'No title')}")
                    print(f"   Blocks: {len(page_result.get('blocks', []))} content blocks")
            else:
                print("   SKIPPED: No page ID found from search")
        else:
            print("   SKIPPED: No pages found to test with")
    except Exception as e:
        print(f"   EXCEPTION: {str(e)}")
    
    # Test 3: Search for databases
    print("\n3. Testing search_pages with database filter...")
    try:
        db_results = notion_tool.search_pages(filter="database", page_size=5)
        if 'error' in db_results:
            print(f"   ERROR: {db_results['error']}")
        else:
            print(f"   SUCCESS: Found {len(db_results)} databases")
            if db_results:
                print(f"   First database ID: {db_results[0].get('id', 'N/A')}")
    except Exception as e:
        print(f"   EXCEPTION: {str(e)}")
    
    # Test 4: Get database (if we found one)
    print("\n4. Testing get_database...")
    try:
        # First, search for a database
        db_results = notion_tool.search_pages(filter="database", page_size=1)
        if 'error' not in db_results and db_results:
            db_id = db_results[0].get('id')
            if db_id:
                db_result = notion_tool.get_database(db_id)
                if 'error' in db_result:
                    print(f"   ERROR: {db_result['error']}")
                else:
                    print(f"   SUCCESS: Retrieved database {db_id}")
                    print(f"   Title: {db_result.get('title_text', 'No title')}")
                    props = db_result.get('properties', {})
                    print(f"   Properties: {len(props)} property types")
                    for prop_name, prop_data in list(props.items())[:3]:
                        prop_type = prop_data.get('type', 'unknown') if isinstance(prop_data, dict) else 'unknown'
                        print(f"     - {prop_name}: {prop_type}")
            else:
                print("   SKIPPED: No database ID found from search")
        else:
            print("   SKIPPED: No databases found to test with")
    except Exception as e:
        print(f"   EXCEPTION: {str(e)}")
    
    # Test 5: Query database (if we found one)
    print("\n5. Testing query_database...")
    try:
        # First, search for a database
        db_results = notion_tool.search_pages(filter="database", page_size=1)
        if 'error' not in db_results and db_results:
            db_id = db_results[0].get('id')
            if db_id:
                query_result = notion_tool.query_database(db_id, page_size=3)
                if 'error' in query_result:
                    print(f"   ERROR: {query_result['error']}")
                else:
                    results = query_result.get('results', [])
                    print(f"   SUCCESS: Queried database {db_id}")
                    print(f"   Found {len(results)} entries")
                    print(f"   Has more: {query_result.get('has_more', False)}")
            else:
                print("   SKIPPED: No database ID found from search")
        else:
            print("   SKIPPED: No databases found to test with")
    except Exception as e:
        print(f"   EXCEPTION: {str(e)}")
    
    # Test 6: Create a test page
    print("\n6. Testing create_page...")
    try:
        # First, find a parent page to create under
        search_results = notion_tool.search_pages(page_size=1)
        if 'error' not in search_results and search_results:
            parent_id = search_results[0].get('id')
            if parent_id:
                test_title = f"Test Page - {int(time.time())}"
                create_result = notion_tool.create_page(
                    parent_id=parent_id,
                    title=test_title,
                    parent_type="page_id"
                )
                if 'error' in create_result:
                    print(f"   ERROR: {create_result['error']}")
                else:
                    print(f"   SUCCESS: Created page '{create_result.get('title', 'N/A')}'")
                    print(f"   Page ID: {create_result.get('id', 'N/A')}")
                    print(f"   URL: {create_result.get('url', 'N/A')}")
            else:
                print("   SKIPPED: No parent page ID found")
        else:
            print("   SKIPPED: No pages found to use as parent")
    except Exception as e:
        print(f"   EXCEPTION: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test 7: Test tool properties
    print("\n7. Testing tool properties...")
    try:
        props = [
            notion_tool.search_pages_property,
            notion_tool.get_page_property,
            notion_tool.create_page_property,
            notion_tool.update_page_property,
            notion_tool.archive_page_property,
            notion_tool.query_database_property,
            notion_tool.get_database_property,
            notion_tool.create_database_entry_property,
            notion_tool.update_database_entry_property
        ]
        print(f"   SUCCESS: All {len(props)} tool properties are accessible")
        for prop in props:
            tool_name = prop.get('name', 'unknown')
            print(f"     - {tool_name}")
    except Exception as e:
        print(f"   EXCEPTION: {str(e)}")
    
    print("\n" + "=" * 80)
    print("Test completed!")
    print("=" * 80)

if __name__ == "__main__":
    test_notion_tool()

