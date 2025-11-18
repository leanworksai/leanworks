#!/usr/bin/env python3
"""
Test suite for Search Tool Quality Assessment

This test suite evaluates the quality of the search tool implementation,
including date filtering, context formatting, and result completeness.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timezone
from typing import Dict, List, Any

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from leanworks.agent.tools.search import SearchTool
from leanworks.storage.gcs import CloudStorage
from leanworks.secret import GCPSecretLoader
from google.cloud import bigquery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class SearchToolQualityTester:
    """Test suite for evaluating search tool quality"""
    
    def __init__(self):
        self.search_tool = None
        self.test_results = []
        
    async def setup(self):
        """Initialize the search tool for testing"""
        try:
            # Initialize BigQuery client with credentials
            bq_client = bigquery.Client.from_service_account_json("gcp_credential.json")
            
            # Initialize storage and secret clients
            storage_client = CloudStorage("gcp_credential.json", client_domain="leanworks.ai")
            secret_client = GCPSecretLoader("gcp_credential.json")
            
            # Initialize search tool
            self.search_tool = SearchTool(
                storage_client=storage_client,
                secret_client=secret_client,
                read_document_ids=set()
            )
            
            logger.info("✅ Search tool initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize search tool: {e}")
            return False
    
    def evaluate_context_quality(self, context: str, test_name: str) -> Dict[str, Any]:
        """Evaluate the quality of formatted context"""
        issues = []
        quality_score = 100
        
        # Check for empty or minimal context
        if not context or len(context.strip()) < 50:
            issues.append("Context is too short or empty")
            quality_score -= 30
        
        # Check for incomplete document headers
        if "DOCUMENT - Date: , Source:" in context:
            issues.append("Incomplete document headers (missing dates)")
            quality_score -= 20
        
        # Check for truncated content
        if context.count("...") > 2:
            issues.append("Excessive content truncation")
            quality_score -= 15
        
        # Check for proper date formatting
        if "Date: " in context and "from " not in context:
            issues.append("Missing date information in context")
            quality_score -= 10
        
        # Check for source information
        if "Source: " not in context:
            issues.append("Missing source information")
            quality_score -= 10
        
        # Check for document IDs
        if "Doc ID:" not in context:
            issues.append("Missing document IDs")
            quality_score -= 5
        
        # Check for proper formatting structure
        if context.count("DOCUMENT -") == 0:
            issues.append("No properly formatted documents found")
            quality_score -= 25
        
        return {
            "test_name": test_name,
            "quality_score": max(0, quality_score),
            "issues": issues,
            "context_length": len(context),
            "document_count": context.count("DOCUMENT -")
        }
    
    async def test_basic_search(self) -> Dict[str, Any]:
        """Test basic search functionality without date filters"""
        logger.info("🔍 Testing basic search functionality...")
        
        try:
            result = self.search_tool.search_documents(
                query="project updates",
                data_source="github_commits"
            )
            
            context = str(result)
            quality = self.evaluate_context_quality(context, "basic_search")
            
            return {
                "test": "basic_search",
                "success": True,
                "quality": quality,
                "has_data_sources": len(result._search_data_sources) > 0,
                "data_source_count": len(result._search_data_sources)
            }
            
        except Exception as e:
            logger.error(f"❌ Basic search test failed: {e}")
            return {
                "test": "basic_search",
                "success": False,
                "error": str(e)
            }
    
    async def test_date_filtering(self) -> Dict[str, Any]:
        """Test date filtering functionality"""
        logger.info("📅 Testing date filtering functionality...")
        
        try:
            # Test with date range - use more recent dates that are likely to have data
            start_date = "2024-01-01"
            end_date = "2025-12-31"
            
            result = self.search_tool.search_documents(
                query="backend improvements",
                data_source="github_commits",
                start_date=start_date,
                end_date=end_date
            )
            
            context = str(result)
            quality = self.evaluate_context_quality(context, "date_filtering")
            
            # Check if date filtering was applied (should see timestamp logs)
            return {
                "test": "date_filtering",
                "success": True,
                "quality": quality,
                "start_date": start_date,
                "end_date": end_date,
                "has_data_sources": len(result._search_data_sources) > 0
            }
            
        except Exception as e:
            logger.error(f"❌ Date filtering test failed: {e}")
            return {
                "test": "date_filtering",
                "success": False,
                "error": str(e)
            }
    
    async def test_no_date_filtering(self) -> Dict[str, Any]:
        """Test that no date filtering is applied when no dates provided"""
        logger.info("🚫 Testing no date filtering...")
        
        try:
            result = self.search_tool.search_documents(
                query="recent changes",
                data_source="github_commits"
                # No start_date or end_date provided
            )
            
            context = str(result)
            quality = self.evaluate_context_quality(context, "no_date_filtering")
            
            return {
                "test": "no_date_filtering",
                "success": True,
                "quality": quality,
                "has_data_sources": len(result._search_data_sources) > 0
            }
            
        except Exception as e:
            logger.error(f"❌ No date filtering test failed: {e}")
            return {
                "test": "no_date_filtering",
                "success": False,
                "error": str(e)
            }
    
    async def test_date_conversion(self) -> Dict[str, Any]:
        """Test date string to timestamp conversion"""
        logger.info("🕐 Testing date conversion...")
        
        test_cases = [
            ("2024-01-01", "Date only format"),
            ("2024-01-01T12:00:00Z", "ISO format with Z"),
            ("2024-01-01T12:00:00+00:00", "ISO format with timezone"),
            ("invalid-date", "Invalid date format"),
            ("", "Empty date string"),
            (None, "None date")
        ]
        
        results = []
        for date_str, description in test_cases:
            try:
                timestamp = self.search_tool._convert_date_to_timestamp(date_str)
                results.append({
                    "input": date_str,
                    "description": description,
                    "timestamp": timestamp,
                    "success": timestamp is not None or date_str in ["invalid-date", "", None]
                })
            except Exception as e:
                results.append({
                    "input": date_str,
                    "description": description,
                    "error": str(e),
                    "success": False
                })
        
        return {
            "test": "date_conversion",
            "success": all(r["success"] for r in results),
            "results": results
        }
    
    async def test_timestamp_debugging(self) -> Dict[str, Any]:
        """Test to debug timestamp issues in the database"""
        logger.info("🔍 Testing timestamp debugging...")
        
        try:
            # First, get some results without date filtering to see what timestamps exist
            result = self.search_tool.search_documents(
                query="recent commits",
                data_source="github_commits"
            )
            
            context = str(result)
            
            # Look for timestamp patterns in the context
            import re
            timestamp_patterns = re.findall(r'date is (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)', context)
            unix_timestamp_patterns = re.findall(r'timestamp[:\s]+(\d{10,13}(?:\.\d+)?)', context)
            
            return {
                "test": "timestamp_debugging",
                "success": True,
                "iso_timestamps_found": timestamp_patterns,
                "unix_timestamps_found": unix_timestamp_patterns,
                "context_length": len(context),
                "has_timestamps": len(timestamp_patterns) > 0 or len(unix_timestamp_patterns) > 0
            }
            
        except Exception as e:
            logger.error(f"❌ Timestamp debugging test failed: {e}")
            return {
                "test": "timestamp_debugging",
                "success": False,
                "error": str(e)
            }
    
    async def test_context_formatting(self) -> Dict[str, Any]:
        """Test context formatting quality"""
        logger.info("📝 Testing context formatting...")
        
        try:
            result = self.search_tool.search_documents(
                query="test formatting",
                data_source="github_commits"
            )
            
            context = str(result)
            
            # Detailed formatting analysis
            formatting_issues = []
            
            # Check for proper document structure
            if not context.startswith("DOCUMENT -"):
                formatting_issues.append("Context doesn't start with proper document header")
            
            # Check for consistent formatting
            lines = context.split('\n')
            doc_headers = [line for line in lines if line.startswith("DOCUMENT -")]
            
            if len(doc_headers) == 0:
                formatting_issues.append("No document headers found")
            else:
                # Check header consistency
                for header in doc_headers:
                    if "Date:" not in header or "Source:" not in header or "Doc ID:" not in header:
                        formatting_issues.append(f"Incomplete header: {header}")
            
            # Check for content after headers
            content_lines = [line for line in lines if not line.startswith("DOCUMENT -") and line.strip()]
            if len(content_lines) == 0:
                formatting_issues.append("No content found after document headers")
            
            # Check for proper spacing
            if context.count('\n\n') < len(doc_headers) - 1:
                formatting_issues.append("Insufficient spacing between documents")
            
            quality_score = max(0, 100 - len(formatting_issues) * 10)
            
            return {
                "test": "context_formatting",
                "success": len(formatting_issues) == 0,
                "quality_score": quality_score,
                "issues": formatting_issues,
                "document_count": len(doc_headers),
                "content_lines": len(content_lines),
                "context_length": len(context)
            }
            
        except Exception as e:
            logger.error(f"❌ Context formatting test failed: {e}")
            return {
                "test": "context_formatting",
                "success": False,
                "error": str(e)
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all quality tests"""
        logger.info("🚀 Starting Search Tool Quality Assessment...")
        
        if not await self.setup():
            return {"error": "Failed to initialize search tool"}
        
        tests = [
            self.test_basic_search(),
            self.test_date_filtering(),
            self.test_no_date_filtering(),
            self.test_date_conversion(),
            self.test_timestamp_debugging(),
            self.test_context_formatting()
        ]
        
        results = await asyncio.gather(*tests, return_exceptions=True)
        
        # Process results
        test_results = []
        total_score = 0
        passed_tests = 0
        
        for result in results:
            if isinstance(result, Exception):
                test_results.append({
                    "test": "unknown",
                    "success": False,
                    "error": str(result)
                })
            else:
                test_results.append(result)
                if result.get("success", False):
                    passed_tests += 1
                    if "quality" in result:
                        total_score += result["quality"]["quality_score"]
                    elif "quality_score" in result:
                        total_score += result["quality_score"]
        
        avg_quality = total_score / len([r for r in test_results if r.get("success", False) and ("quality" in r or "quality_score" in r)]) if passed_tests > 0 else 0
        
        return {
            "summary": {
                "total_tests": len(tests),
                "passed_tests": passed_tests,
                "failed_tests": len(tests) - passed_tests,
                "average_quality_score": avg_quality,
                "overall_success": passed_tests == len(tests)
            },
            "test_results": test_results
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print test results in a formatted way"""
        print("\n" + "="*80)
        print("🔍 SEARCH TOOL QUALITY ASSESSMENT RESULTS")
        print("="*80)
        
        if "error" in results:
            print(f"❌ Setup Error: {results['error']}")
            return
        
        summary = results["summary"]
        print(f"📊 Test Summary:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']} ✅")
        print(f"   Failed: {summary['failed_tests']} ❌")
        print(f"   Average Quality Score: {summary['average_quality_score']:.1f}/100")
        print(f"   Overall Success: {'✅ PASS' if summary['overall_success'] else '❌ FAIL'}")
        
        print(f"\n📋 Detailed Results:")
        for result in results["test_results"]:
            test_name = result.get("test", "unknown")
            success = result.get("success", False)
            status = "✅ PASS" if success else "❌ FAIL"
            
            print(f"\n   {status} {test_name.upper()}")
            
            if not success:
                print(f"      Error: {result.get('error', 'Unknown error')}")
            else:
                if "quality" in result:
                    quality = result["quality"]
                    print(f"      Quality Score: {quality['quality_score']}/100")
                    if quality["issues"]:
                        print(f"      Issues: {', '.join(quality['issues'])}")
                    print(f"      Context Length: {quality['context_length']} chars")
                    print(f"      Documents: {quality['document_count']}")
                
                if "quality_score" in result:
                    print(f"      Quality Score: {result['quality_score']}/100")
                    if result.get("issues"):
                        print(f"      Issues: {', '.join(result['issues'])}")
                
                if "has_data_sources" in result:
                    print(f"      Data Sources: {result['data_source_count'] if 'data_source_count' in result else 'Unknown'}")
        
        print("\n" + "="*80)

async def main():
    """Main test runner"""
    tester = SearchToolQualityTester()
    results = await tester.run_all_tests()
    tester.print_results(results)
    
    # Return exit code based on success
    if "error" in results:
        return 1
    elif not results["summary"]["overall_success"]:
        return 1
    else:
        return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
