#!/usr/bin/env python3
"""
Demonstration of SearchTool functionality - shows how search queries are processed.

This demo uses mock data to illustrate the SearchTool's behavior without requiring
external services that cause segmentation faults on macOS ARM64.
"""

from datetime import datetime, timezone


class SearchResult:
    """SearchResult class for demonstration."""
    def __init__(self, formatted_context: str, data_sources: list = None):
        self.formatted_context = formatted_context
        self._search_data_sources = data_sources or []

    def __str__(self):
        return self.formatted_context

    def __repr__(self):
        return f"SearchResult(context_length={len(self.formatted_context)}, sources={len(self._search_data_sources)})"

    def __contains__(self, item):
        return item in self.formatted_context

    def __len__(self):
        return len(self.formatted_context)


class DemoSearchTool:
    """Demonstration SearchTool that shows how queries are processed."""

    def __init__(self):
        self.org_slug = "demo-org"
        print("🔧 Initialized Demo SearchTool")

    def _convert_date_to_timestamp(self, date_str: str):
        """Convert date string to timestamp."""
        if not date_str:
            return None
        try:
            if 'T' in date_str:
                if date_str.endswith('Z'):
                    date_str = date_str[:-1] + '+00:00'
                dt = datetime.fromisoformat(date_str)
            else:
                dt = datetime.strptime(date_str, '%Y-%m-%d')

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except (ValueError, TypeError):
            return None

    def _build_timestamp_filter(self, start_date=None, end_date=None):
        """Build timestamp filter."""
        filters = {}
        if start_date:
            start_ts = self._convert_date_to_timestamp(start_date)
            if start_ts:
                filters["$gte"] = start_ts
        if end_date:
            end_ts = self._convert_date_to_timestamp(end_date)
            if end_ts:
                filters["$lte"] = end_ts
        return filters

    def _convert_unix_timestamps_in_text(self, text: str) -> str:
        """Convert Unix timestamps in text to readable format."""
        import re

        def replace_timestamp(match):
            timestamp_str = match.group(2)
            try:
                unix_timestamp = float(timestamp_str)
                if unix_timestamp > 1e10:  # milliseconds
                    unix_timestamp = unix_timestamp / 1000
                dt = datetime.fromtimestamp(unix_timestamp, tz=timezone.utc)
                iso_format = dt.isoformat()
                return f"{match.group(1)}{iso_format}"
            except (ValueError, OSError):
                return match.group(0)

        unix_pattern = r'(\w*[Tt]imestamp\s*is\s*)(\d{10,13}(?:\.\d+)?)'
        return re.sub(unix_pattern, replace_timestamp, text)

    def _mock_search_knowledge_base(self, query, filters=None):
        """Mock knowledge base search results."""
        print(f"🔍 Searching knowledge base for: '{query}'")

        if filters:
            print(f"   📅 Applied filters: {filters}")

        # Mock search results
        mock_results = [
            {
                "doc_id": "doc_001",
                "data_source": "github_commits",
                "context": f"This commit fixes a bug in the search functionality. Query processed: {query}",
                "timestamp": 1640995200,  # 2022-01-01
                "metadata": {"author": "developer@example.com", "repo": "leanworks/search"}
            },
            {
                "doc_id": "doc_002",
                "data_source": "jira",
                "context": f"Ticket: Implement advanced search features. The search query '{query}' should return relevant results.",
                "timestamp": 1641081600,  # 2022-01-02
                "metadata": {"priority": "high", "assignee": "product@example.com"}
            },
            {
                "doc_id": "doc_003",
                "data_source": "confluence",
                "context": f"Documentation: Search Tool API. Use this tool when you need to find information about {query}.",
                "timestamp": 1641168000,  # 2022-01-03
                "metadata": {"space": "engineering", "author": "tech-writer@example.com"}
            }
        ]

        # Apply filters if provided
        if filters:
            filtered_results = []
            for result in mock_results:
                include = True
                if "$gte" in filters and result["timestamp"] < filters["$gte"]:
                    include = False
                if "$lte" in filters and result["timestamp"] > filters["$lte"]:
                    include = False
                if include:
                    filtered_results.append(result)
            mock_results = filtered_results

        print(f"   📊 Found {len(mock_results)} relevant documents")
        return mock_results

    def _mock_query_rewrites(self, query):
        """Mock query rewriting."""
        print(f"🔄 Rewriting query: '{query}'")

        rewrites = [
            f"{query} implementation",
            f"{query} documentation",
            f"how to {query}",
            f"{query} best practices"
        ]

        print(f"   📝 Generated {len(rewrites)} query variants:")
        for i, rewrite in enumerate(rewrites, 1):
            print(f"      {i}. '{rewrite}'")

        return rewrites

    def _format_results(self, results):
        """Format search results like the real SearchTool."""
        formatted_context = ""

        for result in results:
            # Convert timestamp to readable format
            dt = datetime.fromtimestamp(result["timestamp"], tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")

            # Format document header
            title = f"DOCUMENT - Date: {date_str}, Source: {result['data_source']}, Doc ID: {result['doc_id']}"

            # Process context (convert any timestamps)
            context = self._convert_unix_timestamps_in_text(result["context"])

            formatted_context += f"{title}\n{context}\n\n"

        return formatted_context, [r["data_source"] for r in results]

    def search_documents(self, query, data_source=None, start_date=None, end_date=None,
                        search_scope="all", tool_name=None):
        """Demonstrate search document processing."""

        print(f"\n{'='*80}")
        print(f"🔎 SEARCH TOOL DEMONSTRATION")
        print(f"{'='*80}")
        print(f"Query: '{query}'")
        print(f"Scope: {search_scope}")
        if data_source:
            print(f"Data Source: {data_source}")
        if start_date or end_date:
            print(f"Date Range: {start_date or 'any'} to {end_date or 'any'}")
        print()

        # Step 1: Query rewriting
        rewrites = self._mock_query_rewrites(query)
        all_queries = [query] + rewrites

        # Step 2: Build filters
        filters = self._build_timestamp_filter(start_date, end_date)
        if data_source:
            filters["data_source"] = {"$eq": data_source}

        # Step 3: Search execution
        results = self._mock_search_knowledge_base(query, filters)

        # Step 4: Format results
        formatted_context, data_sources = self._format_results(results)

        print(f"\n📄 FORMATTED RESULTS:")
        print(f"{'-'*40}")
        print(formatted_context)

        print(f"📊 SUMMARY:")
        print(f"   • Documents found: {len(results)}")
        print(f"   • Data sources: {', '.join(set(data_sources))}")
        print(f"   • Total context length: {len(formatted_context)} characters")
        print(f"{'='*80}\n")

        return SearchResult(formatted_context, data_sources)


def main():
    """Demonstrate SearchTool with sample queries."""
    print("🚀 SearchTool Demonstration")
    print("This shows how the SearchTool processes queries and formats results.\n")

    search_tool = DemoSearchTool()

    # Sample queries to demonstrate
    queries = [
        ("implement search functionality", None, None, None),
        ("API documentation", "confluence", None, None),
        ("bug fixes", "github_commits", "2022-01-01", "2022-01-15"),
        ("user authentication", None, "2022-01-01", None),
    ]

    for query, data_source, start_date, end_date in queries:
        result = search_tool.search_documents(
            query=query,
            data_source=data_source,
            start_date=start_date,
            end_date=end_date
        )

        # Demonstrate SearchResult usage
        print(f"💡 SearchResult object properties:")
        print(f"   • String length: {len(result)} characters")
        print(f"   • Contains 'search': {'search' in result}")
        print(f"   • Data sources: {result._search_data_sources}")
        print(f"   • Repr: {repr(result)}")
        print()

        # Continue to next query automatically


if __name__ == "__main__":
    main()