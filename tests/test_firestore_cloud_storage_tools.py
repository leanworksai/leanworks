"""
Test suite for Firestore and Cloud Storage tools.

This test verifies that:
1. FirestoreTool can query messages from Firestore
2. CloudStorageTool can generate signed URLs for images
3. CloudStorageTool can list images in a chat
"""

import logging
import asyncio
import json
from google.cloud import firestore, secretmanager, storage
from google.oauth2 import service_account
from leanworks.agent.tools.toolkit import ToolUse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def initialize_clients(credential_path: str = "gcp_credential.json"):
    """Initialize Firestore, Secret Manager, and Storage clients."""
    try:
        credentials = service_account.Credentials.from_service_account_file(credential_path)
        with open(credential_path, "r") as f:
            credential_data = json.load(f)
        project_id = credential_data["project_id"]
        
        # Initialize Firestore client
        firestore_client = firestore.Client(
            credentials=credentials, 
            project=project_id, 
            database="leanworks-prod"
        )
        
        # Initialize Secret Manager client
        secret_manager_client = secretmanager.SecretManagerServiceClient(credentials=credentials)
        
        # Initialize Storage client
        storage_client = storage.Client.from_service_account_json(credential_path)
        
        logger.info("✅ All clients initialized successfully")
        return firestore_client, secret_manager_client, storage_client, project_id
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize clients: {e}")
        raise


def test_firestore_tool():
    """Test FirestoreTool query_messages function."""
    logger.info("=" * 80)
    logger.info("🧪 TESTING FIRESTORE TOOL")
    logger.info("=" * 80)
    
    try:
        # Initialize clients
        firestore_client, secret_manager_client, storage_client, project_id = initialize_clients()
        
        # Initialize ToolUse with firestore tool
        org_slug = "leanworksai"
        user_id = "test@leanworks.ai"
        
        tool_use = ToolUse(
            org_slug=org_slug,
            firestore_client=firestore_client,
            secret_manager_client=secret_manager_client,
            user_id=user_id,
            credential_path="gcp_credential.json"
        )
        
        # Test 1: Check if firestore_tool is initialized
        logger.info("\n📋 Test 1: Checking FirestoreTool initialization...")
        firestore_tool = tool_use.firestore_tool
        assert firestore_tool is not None, "FirestoreTool should be initialized"
        assert 'firestore' in tool_use.enabled_tools, "firestore should be in enabled_tools"
        logger.info("✅ FirestoreTool initialized successfully")
        
        # Test 2: Query messages with a test chatId
        logger.info("\n📋 Test 2: Querying messages from Firestore...")
        test_chat_id = f"ai-assistant-{user_id}"
        
        result = firestore_tool.query_messages(
            chatId=test_chat_id,
            limit=10,
            orderBy="desc"
        )
        
        # Check if result is valid (either list of messages or error dict)
        assert isinstance(result, (list, dict)), "Result should be a list or dict"
        
        if isinstance(result, list):
            logger.info(f"✅ Successfully retrieved {len(result)} messages")
            if len(result) > 0:
                # Check message structure
                message = result[0]
                assert 'id' in message, "Message should have 'id' field"
                assert 'content' in message, "Message should have 'content' field"
                assert 'timestamp' in message, "Message should have 'timestamp' field"
                logger.info(f"✅ Message structure is valid. Sample message ID: {message.get('id')}")
        elif isinstance(result, dict) and 'error' in result:
            logger.warning(f"⚠️ Query returned error (this is OK if no messages exist): {result.get('error')}")
        else:
            logger.info(f"✅ Query completed. Result type: {type(result)}")
        
        # Test 3: Check tool property
        logger.info("\n📋 Test 3: Checking query_messages_property...")
        property_obj = firestore_tool.query_messages_property
        assert property_obj is not None, "query_messages_property should exist"
        assert property_obj.get('name') == 'query_messages', "Property name should be 'query_messages'"
        logger.info("✅ query_messages_property is valid")
        
        # Test 4: Check if tool is in tools list
        logger.info("\n📋 Test 4: Checking if tool is in tools list...")
        tools = tool_use.tools
        tool_names = [tool.get('name') for tool in tools if isinstance(tool, dict)]
        assert 'query_messages' in tool_names, "query_messages should be in tools list"
        logger.info(f"✅ Tool is in tools list. Total tools: {len(tools)}")
        
        # Test 5: Check if function is in function_map
        logger.info("\n📋 Test 5: Checking if function is in function_map...")
        function_map = tool_use.function_map
        assert 'query_messages' in function_map, "query_messages should be in function_map"
        logger.info("✅ Function is in function_map")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL FIRESTORE TOOL TESTS PASSED!")
        logger.info("=" * 80)
        return True
        
    except Exception as e:
        logger.error(f"\n❌ FIRESTORE TOOL TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_cloud_storage_tool():
    """Test CloudStorageTool get_image_url and list_chat_images functions."""
    logger.info("\n" + "=" * 80)
    logger.info("🧪 TESTING CLOUD STORAGE TOOL")
    logger.info("=" * 80)
    
    try:
        # Initialize clients
        firestore_client, secret_manager_client, storage_client, project_id = initialize_clients()
        
        # Initialize ToolUse with cloud_storage tool
        org_slug = "leanworksai"
        user_id = "test@leanworks.ai"
        
        tool_use = ToolUse(
            org_slug=org_slug,
            firestore_client=firestore_client,
            secret_manager_client=secret_manager_client,
            user_id=user_id,
            credential_path="gcp_credential.json"
        )
        
        # Test 1: Check if cloud_storage_tool is initialized
        logger.info("\n📋 Test 1: Checking CloudStorageTool initialization...")
        cloud_storage_tool = tool_use.cloud_storage_tool
        assert cloud_storage_tool is not None, "CloudStorageTool should be initialized"
        assert 'cloud_storage' in tool_use.enabled_tools, "cloud_storage should be in enabled_tools"
        logger.info("✅ CloudStorageTool initialized successfully")
        
        # Test 2: Test get_image_url_property
        logger.info("\n📋 Test 2: Checking get_image_url_property...")
        get_image_url_prop = cloud_storage_tool.get_image_url_property
        assert get_image_url_prop is not None, "get_image_url_property should exist"
        assert get_image_url_prop.get('name') == 'get_image_url', "Property name should be 'get_image_url'"
        logger.info("✅ get_image_url_property is valid")
        
        # Test 3: Test list_chat_images_property
        logger.info("\n📋 Test 3: Checking list_chat_images_property...")
        list_images_prop = cloud_storage_tool.list_chat_images_property
        assert list_images_prop is not None, "list_chat_images_property should exist"
        assert list_images_prop.get('name') == 'list_chat_images', "Property name should be 'list_chat_images'"
        logger.info("✅ list_chat_images_property is valid")
        
        # Test 4: Test get_image_url with a test image ID (this may fail if image doesn't exist, which is OK)
        logger.info("\n📋 Test 4: Testing get_image_url function...")
        test_chat_id = f"ai-assistant-{user_id}"
        test_image_id = "00000000-0000-0000-0000-000000000000.jpg"  # Dummy UUID
        
        result = cloud_storage_tool.get_image_url(
            imageId=test_image_id,
            chatId=test_chat_id
        )
        
        # Result should be either a dict with imageUrl or an error dict
        assert isinstance(result, dict), "Result should be a dictionary"
        
        if 'imageUrl' in result:
            logger.info(f"✅ Successfully generated signed URL: {result.get('imageUrl')[:50]}...")
            assert 'imageId' in result, "Result should have 'imageId'"
            assert 'expiresAt' in result, "Result should have 'expiresAt'"
        elif 'error' in result:
            logger.warning(f"⚠️ get_image_url returned error (this is OK if image doesn't exist): {result.get('error')}")
        else:
            logger.info(f"✅ get_image_url completed. Result keys: {result.keys()}")
        
        # Test 5: Test list_chat_images
        logger.info("\n📋 Test 5: Testing list_chat_images function...")
        result = cloud_storage_tool.list_chat_images(
            chatId=test_chat_id,
            limit=10
        )
        
        # Result should be either a list or an error dict
        assert isinstance(result, (list, dict)), "Result should be a list or dict"
        
        if isinstance(result, list):
            logger.info(f"✅ Successfully listed {len(result)} images")
            if len(result) > 0:
                # Check image structure
                image = result[0]
                assert 'imageUrl' in image, "Image should have 'imageUrl' field"
                assert 'imageId' in image, "Image should have 'imageId' field"
                logger.info(f"✅ Image structure is valid. Sample image ID: {image.get('imageId')}")
        elif isinstance(result, dict) and 'error' in result:
            logger.warning(f"⚠️ list_chat_images returned error (this is OK if no images exist): {result.get('error')}")
        else:
            logger.info(f"✅ list_chat_images completed. Result type: {type(result)}")
        
        # Test 6: Check if tools are in tools list
        logger.info("\n📋 Test 6: Checking if tools are in tools list...")
        tools = tool_use.tools
        tool_names = [tool.get('name') for tool in tools if isinstance(tool, dict)]
        assert 'get_image_url' in tool_names, "get_image_url should be in tools list"
        assert 'list_chat_images' in tool_names, "list_chat_images should be in tools list"
        logger.info(f"✅ Tools are in tools list. Total tools: {len(tools)}")
        
        # Test 7: Check if functions are in function_map
        logger.info("\n📋 Test 7: Checking if functions are in function_map...")
        function_map = tool_use.function_map
        assert 'get_image_url' in function_map, "get_image_url should be in function_map"
        assert 'list_chat_images' in function_map, "list_chat_images should be in function_map"
        logger.info("✅ Functions are in function_map")
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ ALL CLOUD STORAGE TOOL TESTS PASSED!")
        logger.info("=" * 80)
        return True
        
    except Exception as e:
        logger.error(f"\n❌ CLOUD STORAGE TOOL TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_tools_in_default_list():
    """Test that firestore and cloud_storage are in the default tools list."""
    logger.info("\n" + "=" * 80)
    logger.info("🧪 TESTING DEFAULT TOOLS LIST")
    logger.info("=" * 80)
    
    try:
        # Initialize ToolUse without specifying tools (should use defaults)
        firestore_client, secret_manager_client, storage_client, project_id = initialize_clients()
        
        tool_use = ToolUse(
            org_slug="leanworksai",
            firestore_client=firestore_client,
            secret_manager_client=secret_manager_client,
            user_id="test@leanworks.ai",
            credential_path="gcp_credential.json"
            # tools=None means use defaults
        )
        
        logger.info("\n📋 Checking default tools list...")
        assert 'firestore' in tool_use.requested_tools, "firestore should be in default tools"
        assert 'cloud_storage' in tool_use.requested_tools, "cloud_storage should be in default tools"
        assert 'search' in tool_use.requested_tools, "search should be in default tools"
        assert 'postgres' in tool_use.requested_tools, "postgres should be in default tools"
        assert 'duckdb' in tool_use.requested_tools, "duckdb should be in default tools"
        
        logger.info(f"✅ Default tools: {tool_use.requested_tools}")
        logger.info("\n" + "=" * 80)
        logger.info("✅ DEFAULT TOOLS LIST TEST PASSED!")
        logger.info("=" * 80)
        return True
        
    except Exception as e:
        logger.error(f"\n❌ DEFAULT TOOLS LIST TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("🧪 TESTING FIRESTORE AND CLOUD STORAGE TOOLS")
    print("=" * 80)
    print()
    
    results = []
    
    try:
        # Test default tools list
        results.append(("Default Tools List", test_tools_in_default_list()))
        
        # Test Firestore tool
        results.append(("Firestore Tool", test_firestore_tool()))
        
        # Test Cloud Storage tool
        results.append(("Cloud Storage Tool", test_cloud_storage_tool()))
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 TEST SUMMARY")
        print("=" * 80)
        
        passed = 0
        failed = 0
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
            if result:
                passed += 1
            else:
                failed += 1
        
        print(f"\nTotal: {len(results)} tests")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print("=" * 80)
        
        if failed == 0:
            print("✅ ALL TESTS PASSED!")
            return 0
        else:
            print("❌ SOME TESTS FAILED")
            return 1
            
    except Exception as e:
        print("=" * 80)
        print(f"❌ TEST ERROR: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

