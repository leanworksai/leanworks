"""
Data Source Formatter Module

This module provides a scalable and flexible way to format data sources
from various formats and types. It uses a plugin-based approach to handle
different data source formats dynamically.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DataSourceInfo:
    """Represents formatted data source information."""
    display_name: str
    source_type: str
    original_source: str
    metadata: Dict[str, Any] = None

class DataSourceFormatter:
    """
    A scalable formatter for data sources that can handle various formats.
    Uses a plugin-based approach for extensibility.
    """
    
    def __init__(self):
        self.formatters = {
            # Atlassian
            'confluence': self._format_confluence_source,
            'jira': self._format_jira_source,
            
            # GitHub
            'github_commits': self._format_github_commits_source,
            'github_codes': self._format_github_codes_source,
            'github': self._format_github_source,  # fallback
            
            # GitLab
            'gitlab_commits': self._format_gitlab_commits_source,
            'gitlab_codes': self._format_gitlab_codes_source,
            'gitlab_issue': self._format_gitlab_issue_source,
            'gitlab': self._format_gitlab_source,  # fallback
            
            # Google Workspace
            'google_doc': self._format_google_doc_source,
            'google_sheet': self._format_google_sheet_source,
            
            # Collaboration
            'notion': self._format_notion_source,
            'slack': self._format_slack_source,
            'teams': self._format_teams_source,
            
            # Microsoft
            'outlook_email': self._format_outlook_email_source,
            'calendar': self._format_calendar_source,
            
            # ITSM
            'servicenow': self._format_servicenow_source,
            
            # Storage
            'meeting_notes': self._format_meeting_notes_source,
            
            # Legacy/Generic
            'api_docs': self._format_api_docs_source,
            'system_docs': self._format_system_docs_source,
            'database': self._format_database_source,
            'url': self._format_url_source,
            'default': self._format_default_source
        }
        
        # Regex patterns for source type detection (ordered by specificity)
        self.detection_patterns = {
            # Atlassian
            'confluence': r'(atlassian\.net.*wiki|confluence)',
            'jira': r'(atlassian\.net.*browse|jira)',
            
            # GitHub specific
            'github_commits': r'github\.com.*commit',
            'github_codes': r'github\.com.*blob',
            'github': r'github\.com',
            
            # GitLab specific
            'gitlab_commits': r'gitlab\..*commit',
            'gitlab_codes': r'gitlab\..*blob',
            'gitlab_issue': r'gitlab\..*issues',
            'gitlab': r'gitlab\.',
            
            # Google Workspace
            'google_doc': r'docs\.google\.com/document',
            'google_sheet': r'docs\.google\.com/spreadsheets',
            
            # Collaboration
            'notion': r'notion\.so',
            'slack': r'^slack/',
            'teams': r'^[^/]+/[^/]+$',  # team-name/channel-name pattern
            
            # Microsoft
            'outlook_email': r'outlook\.office\.com.*mail',
            'calendar': r'(calendar|outlook|meeting)',
            
            # ITSM
            'servicenow': r'service-now\.com',
            
            # Storage
            'meeting_notes': r'^gs://',
            
            # Legacy/Generic
            'api_docs': r'api[_-]?docs?',
            'system_docs': r'system[_-]?docs?',
            'database': r'(bigquery|duckdb|database|db)',
            'url': r'https?://',
        }
    
    def format_data_sources(self, links: List[str], contexts: List[Dict[str, Any]] = None, simple_mode: bool = False, show_all_links: bool = False, raw_links_only: bool = False) -> List[str]:
        """
        Format data sources from various inputs into a consistent format.
        
        Args:
            links: List of URLs or links
            contexts: List of context dictionaries with metadata
            simple_mode: If True, returns "Data Source Type: Raw Link" format
            show_all_links: If True, extracts and shows all unique links from context data
            raw_links_only: If True, returns just the raw links without any prefixes
            
        Returns:
            List of formatted data source strings
        """
        formatted_sources = []
        seen_sources = set()
        seen_links = set()
        
        # If show_all_links is True, extract all unique links from context data
        if show_all_links and contexts:
            for ctx in contexts:
                context_text = ctx.get("context", "")
                # Extract URLs from context text
                import re
                url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
                urls = re.findall(url_pattern, context_text)
                
                for url in urls:
                    if url not in seen_links:
                        seen_links.add(url)
                        if raw_links_only:
                            # Raw links only: just show the URL without any prefixes
                            formatted_sources.append(url)
                        elif simple_mode:
                            # Simple mode: just show data source type and raw link
                            source_type = self._detect_source_type(url)
                            formatted_sources.append(f"{source_type.replace('_', ' ').title()}: {url}")
                        else:
                            # Full formatting mode
                            formatted_source = self._format_source(url)
                            if formatted_source:
                                formatted_sources.append(formatted_source.display_name)
        
        # Process original links
        for link in links:
            if link and link not in seen_sources:
                seen_sources.add(link)
                if raw_links_only:
                    # Raw links only: just show the URL without any prefixes
                    formatted_sources.append(link)
                elif simple_mode:
                    # Simple mode: just show data source type and raw link
                    source_type = self._detect_source_type(link)
                    formatted_sources.append(f"{source_type.replace('_', ' ').title()}: {link}")
                else:
                    # Full formatting mode
                    formatted_source = self._format_source(link)
                    if formatted_source:
                        formatted_sources.append(formatted_source.display_name)
        
        # Process contexts for additional metadata (only if not showing all links)
        if contexts and not show_all_links:
            for ctx in contexts:
                data_source = ctx.get("data_source")
                if data_source and data_source not in seen_sources:
                    seen_sources.add(data_source)
                    if simple_mode:
                        # Simple mode: just show data source type and raw link
                        source_type = self._detect_source_type(data_source, ctx)
                        formatted_sources.append(f"{source_type.replace('_', ' ').title()}: {data_source}")
                    else:
                        # Full formatting mode
                        formatted_source = self._format_source(data_source, ctx)
                        if formatted_source:
                            formatted_sources.append(formatted_source.display_name)
        
        return formatted_sources
    
    def _format_source(self, source: str, context: Dict[str, Any] = None) -> Optional[DataSourceInfo]:
        """
        Format a single data source using the appropriate formatter.
        
        Args:
            source: The source string to format
            context: Optional context dictionary with additional metadata
            
        Returns:
            DataSourceInfo object or None if formatting fails
        """
        source_lower = source.lower()
        
        # Determine source type
        source_type = self._detect_source_type(source_lower, context)
        
        # Get appropriate formatter
        formatter = self.formatters.get(source_type, self.formatters['default'])
        
        try:
            return formatter(source, context)
        except Exception as e:
            logger.warning(f"Error formatting source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _detect_source_type(self, source: str, context: Dict[str, Any] = None) -> str:
        """
        Detect the type of data source based on patterns and context.
        
        Args:
            source: The source string (lowercase)
            context: Optional context dictionary
            
        Returns:
            Source type string
        """
        # Check context metadata first - if data_source is already a known type, use it directly
        if context:
            data_source = context.get("data_source", "").lower()
            # If data_source is already a known source type, return it directly
            if data_source in self.formatters:
                return data_source
            # Otherwise, try to match against patterns
            for pattern_name, pattern in self.detection_patterns.items():
                if re.search(pattern, data_source):
                    return pattern_name
        
        # Check source string patterns (order matters - more specific patterns first)
        # First check for exact matches that should take priority
        if source.startswith("gs://"):
            return 'meeting_notes'
        if source.startswith("slack/"):
            return 'slack'
        if "/" in source and not source.startswith("http") and not source.startswith("gs://"):
            # Check if it matches teams pattern (team-name/channel-name)
            parts = source.split("/")
            if len(parts) == 2 and not any(char in source for char in [".", ":", "?"]):
                return 'teams'
        
        # Then check regex patterns
        for pattern_name, pattern in self.detection_patterns.items():
            if re.search(pattern, source):
                return pattern_name
        
        return 'default'
    
    def _format_confluence_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format Confluence URLs with page information."""
        try:
            parsed = urlparse(source)
            domain = parsed.netloc
            path_parts = parsed.path.strip("/").split("/")
            
            # Extract space key and page title
            if "spaces" in path_parts and "pages" in path_parts:
                spaces_index = path_parts.index("spaces")
                pages_index = path_parts.index("pages")
                
                if spaces_index + 1 < len(path_parts):
                    space_key = path_parts[spaces_index + 1]
                else:
                    space_key = "unknown"
                
                if pages_index + 1 < len(path_parts):
                    page_id = path_parts[pages_index + 1]
                else:
                    page_id = "unknown"
                
                # Get page title (last part, URL decoded)
                page_title = path_parts[-1].replace("+", " ").replace("%20", " ") if path_parts else "Unknown Page"
                
                display_name = f"Confluence: {space_key} - {page_title}"
                
                return DataSourceInfo(
                    display_name=display_name,
                    source_type="confluence",
                    original_source=source,
                    metadata={"space_key": space_key, "page_id": page_id, "page_title": page_title, "domain": domain}
                )
            else:
                return DataSourceInfo(
                    display_name=f"Confluence: {domain}",
                    source_type="confluence",
                    original_source=source,
                    metadata={"domain": domain}
                )
        except Exception as e:
            logger.warning(f"Error parsing Confluence source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_jira_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format Jira URLs with ticket information."""
        try:
            if "/browse/" in source:
                parts = source.split("/browse/")
                if len(parts) > 1:
                    ticket = parts[1]
                    parsed = urlparse(source)
                    domain = parsed.netloc
                    
                    return DataSourceInfo(
                        display_name=f"Jira Ticket: {ticket}",
                        source_type="jira",
                        original_source=source,
                        metadata={"ticket": ticket, "domain": domain}
                    )
            else:
                parsed = urlparse(source)
                domain = parsed.netloc
                return DataSourceInfo(
                    display_name=f"Jira: {domain}",
                    source_type="jira",
                    original_source=source,
                    metadata={"domain": domain}
                )
        except Exception as e:
            logger.warning(f"Error parsing Jira source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_gitlab_commits_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format GitLab commit URLs."""
        try:
            parsed = urlparse(source)
            domain = parsed.netloc
            path_parts = parsed.path.strip("/").split("/")
            
            if "/commit/" in source:
                # Find the commit part and extract repository info
                commit_index = -1
                for i, part in enumerate(path_parts):
                    if part == "commit":
                        commit_index = i
                        break
                
                if commit_index > 0:
                    # Repository name is the part before "commit" (skip any "-" parts)
                    repo_name = None
                    for i in range(commit_index - 1, -1, -1):
                        if path_parts[i] != "-":
                            repo_name = path_parts[i]
                            break
                    
                    if not repo_name:
                        repo_name = "repository"
                    
                    # Extract commit hash (first 8 characters)
                    commit_hash = path_parts[commit_index + 1][:8] if commit_index + 1 < len(path_parts) else "unknown"
                    
                    display_name = f"GitLab Commit: {repo_name} ({commit_hash})"
                    
                    return DataSourceInfo(
                        display_name=display_name,
                        source_type="gitlab_commits",
                        original_source=source,
                        metadata={"repo": repo_name, "commit": commit_hash, "domain": domain}
                    )
            else:
                return self._format_gitlab_source(source, context)
        except Exception as e:
            logger.warning(f"Error parsing GitLab commit source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_gitlab_codes_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format GitLab code file URLs."""
        try:
            parsed = urlparse(source)
            domain = parsed.netloc
            path_parts = parsed.path.strip("/").split("/")
            
            if "/blob/" in source:
                # Find the blob part and extract repository info
                blob_index = -1
                for i, part in enumerate(path_parts):
                    if part == "blob":
                        blob_index = i
                        break
                
                if blob_index > 0:
                    # Repository name is the part before "blob" (skip any "-" parts)
                    repo_name = None
                    for i in range(blob_index - 1, -1, -1):
                        if path_parts[i] != "-":
                            repo_name = path_parts[i]
                            break
                    
                    if not repo_name:
                        repo_name = "repository"
                    
                    # Extract file path
                    if blob_index + 2 < len(path_parts):
                        file_path = "/".join(path_parts[blob_index + 2:])
                        file_name = path_parts[-1] if path_parts else "unknown"
                        branch = path_parts[blob_index + 1] if blob_index + 1 < len(path_parts) else "unknown"
                    else:
                        file_path = "unknown"
                        file_name = "unknown"
                        branch = "unknown"
                    
                    display_name = f"GitLab Code: {repo_name} - {file_name}"
                    
                    return DataSourceInfo(
                        display_name=display_name,
                        source_type="gitlab_codes",
                        original_source=source,
                        metadata={"repo": repo_name, "branch": branch, "file_path": file_path, "file_name": file_name, "domain": domain}
                    )
            else:
                return self._format_gitlab_source(source, context)
        except Exception as e:
            logger.warning(f"Error parsing GitLab code source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_gitlab_issue_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format GitLab issue URLs."""
        try:
            parsed = urlparse(source)
            domain = parsed.netloc
            path_parts = parsed.path.strip("/").split("/")
            
            if "/issues/" in source:
                # Find the issues part and extract repository info
                issues_index = -1
                for i, part in enumerate(path_parts):
                    if part == "issues":
                        issues_index = i
                        break
                
                if issues_index > 0:
                    # Repository name is the part before "issues" (skip any "-" parts)
                    repo_name = None
                    for i in range(issues_index - 1, -1, -1):
                        if path_parts[i] != "-":
                            repo_name = path_parts[i]
                            break
                    
                    if not repo_name:
                        repo_name = "repository"
                    
                    # Extract issue number
                    issue_number = path_parts[issues_index + 1] if issues_index + 1 < len(path_parts) else "unknown"
                    
                    display_name = f"GitLab Issue: {repo_name} #{issue_number}"
                    
                    return DataSourceInfo(
                        display_name=display_name,
                        source_type="gitlab_issue",
                        original_source=source,
                        metadata={"repo": repo_name, "issue_number": issue_number, "domain": domain}
                    )
            else:
                return self._format_gitlab_source(source, context)
        except Exception as e:
            logger.warning(f"Error parsing GitLab issue source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_gitlab_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format GitLab URLs (fallback)."""
        try:
            parsed = urlparse(source)
            domain = parsed.netloc
            path_parts = parsed.path.strip("/").split("/")
            
            # Get the last meaningful part as repository name
            repo_name = path_parts[-1] if path_parts and path_parts[-1] else "repository"
            
            return DataSourceInfo(
                display_name=f"GitLab: {repo_name}",
                source_type="gitlab",
                original_source=source,
                metadata={"repo": repo_name, "domain": domain}
            )
        except Exception as e:
            logger.warning(f"Error parsing GitLab source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_github_commits_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format GitHub commit URLs."""
        try:
            parsed = urlparse(source)
            domain = parsed.netloc
            path_parts = parsed.path.strip("/").split("/")
            
            if len(path_parts) >= 4 and path_parts[2] == "commit":
                owner = path_parts[0]
                repo = path_parts[1]
                commit_hash = path_parts[3][:8] if len(path_parts[3]) > 8 else path_parts[3]
                
                display_name = f"GitHub Commit: {owner}/{repo} ({commit_hash})"
                
                return DataSourceInfo(
                    display_name=display_name,
                    source_type="github_commits",
                    original_source=source,
                    metadata={"owner": owner, "repo": repo, "commit": commit_hash, "domain": domain}
                )
            else:
                return self._format_github_source(source, context)
        except Exception as e:
            logger.warning(f"Error parsing GitHub commit source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_github_codes_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format GitHub code file URLs."""
        try:
            parsed = urlparse(source)
            domain = parsed.netloc
            path_parts = parsed.path.strip("/").split("/")
            
            if len(path_parts) >= 5 and path_parts[2] == "blob":
                owner = path_parts[0]
                repo = path_parts[1]
                branch = path_parts[3]
                file_path = "/".join(path_parts[4:])
                file_name = path_parts[-1] if path_parts else "unknown"
                
                display_name = f"GitHub Code: {owner}/{repo} - {file_name}"
                
                return DataSourceInfo(
                    display_name=display_name,
                    source_type="github_codes",
                    original_source=source,
                    metadata={"owner": owner, "repo": repo, "branch": branch, "file_path": file_path, "file_name": file_name, "domain": domain}
                )
            else:
                return self._format_github_source(source, context)
        except Exception as e:
            logger.warning(f"Error parsing GitHub code source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_github_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format GitHub URLs (fallback)."""
        try:
            parsed = urlparse(source)
            domain = parsed.netloc
            path_parts = parsed.path.strip("/").split("/")
            
            if len(path_parts) >= 2:
                owner = path_parts[0]
                repo = path_parts[1]
                display_name = f"GitHub: {owner}/{repo}"
                
                return DataSourceInfo(
                    display_name=display_name,
                    source_type="github",
                    original_source=source,
                    metadata={"owner": owner, "repo": repo, "domain": domain}
                )
            else:
                return self._format_default_source(source, context)
        except Exception as e:
            logger.warning(f"Error parsing GitHub source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_url_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format generic URLs."""
        try:
            parsed = urlparse(source)
            domain = parsed.netloc
            
            # Extract meaningful path information
            path_parts = parsed.path.strip("/").split("/")
            if path_parts and path_parts[0]:
                resource = path_parts[0].replace("-", " ").replace("_", " ").title()
                display_name = f"Knowledge base: {resource} ({domain})"
            else:
                display_name = f"Knowledge base: {domain}"
            
            return DataSourceInfo(
                display_name=display_name,
                source_type="url",
                original_source=source,
                metadata={"domain": domain, "path": parsed.path}
            )
        except Exception as e:
            logger.warning(f"Error parsing URL source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_api_docs_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format API documentation sources."""
        # Clean up the source name
        clean_name = source.replace("_", " ").replace("-", " ").title()
        
        return DataSourceInfo(
            display_name=f"API Documentation: {clean_name}",
            source_type="api_docs",
            original_source=source,
            metadata={"original_name": source, "clean_name": clean_name}
        )
    
    def _format_system_docs_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format system documentation sources."""
        clean_name = source.replace("_", " ").replace("-", " ").title()
        
        return DataSourceInfo(
            display_name=f"System Documentation: {clean_name}",
            source_type="system_docs",
            original_source=source,
            metadata={"original_name": source, "clean_name": clean_name}
        )
    
    def _format_database_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format database sources."""
        if "bigquery" in source.lower():
            return DataSourceInfo(
                display_name=f"BigQuery Database: {source}",
                source_type="database",
                original_source=source,
                metadata={"db_type": "bigquery"}
            )
        elif "duckdb" in source.lower():
            return DataSourceInfo(
                display_name=f"DuckDB Database: {source}",
                source_type="database",
                original_source=source,
                metadata={"db_type": "duckdb"}
            )
        else:
            return DataSourceInfo(
                display_name=f"Database: {source}",
                source_type="database",
                original_source=source,
                metadata={"db_type": "unknown"}
            )
    
    def _format_google_doc_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format Google Docs URLs."""
        try:
            parsed = urlparse(source)
            domain = parsed.netloc
            path_parts = parsed.path.strip("/").split("/")
            
            # Extract document ID from path
            if "document" in path_parts and "d" in path_parts:
                doc_index = path_parts.index("d")
                if doc_index + 1 < len(path_parts):
                    doc_id = path_parts[doc_index + 1]
                    display_name = f"Google Doc: {doc_id[:8]}..."
                else:
                    display_name = "Google Doc"
            else:
                display_name = "Google Doc"
            
            return DataSourceInfo(
                display_name=display_name,
                source_type="google_doc",
                original_source=source,
                metadata={"domain": domain, "doc_id": doc_id if 'doc_id' in locals() else None}
            )
        except Exception as e:
            logger.warning(f"Error parsing Google Doc source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_google_sheet_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format Google Sheets URLs."""
        try:
            parsed = urlparse(source)
            domain = parsed.netloc
            path_parts = parsed.path.strip("/").split("/")
            
            # Extract sheet ID from path
            if "spreadsheets" in path_parts and "d" in path_parts:
                doc_index = path_parts.index("d")
                if doc_index + 1 < len(path_parts):
                    sheet_id = path_parts[doc_index + 1]
                    display_name = f"Google Sheet: {sheet_id[:8]}..."
                else:
                    display_name = "Google Sheet"
            else:
                display_name = "Google Sheet"
            
            return DataSourceInfo(
                display_name=display_name,
                source_type="google_sheet",
                original_source=source,
                metadata={"domain": domain, "sheet_id": sheet_id if 'sheet_id' in locals() else None}
            )
        except Exception as e:
            logger.warning(f"Error parsing Google Sheet source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_notion_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format Notion URLs."""
        try:
            parsed = urlparse(source)
            domain = parsed.netloc
            path_parts = parsed.path.strip("/").split("/")
            
            # Extract page ID (last part of path)
            if path_parts:
                page_id = path_parts[-1]
                display_name = f"Notion: {page_id[:8]}..."
            else:
                display_name = "Notion"
            
            return DataSourceInfo(
                display_name=display_name,
                source_type="notion",
                original_source=source,
                metadata={"domain": domain, "page_id": page_id if 'page_id' in locals() else None}
            )
        except Exception as e:
            logger.warning(f"Error parsing Notion source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_slack_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format Slack channel references."""
        try:
            # Slack format: "slack/channel-name"
            if source.startswith("slack/"):
                channel = source[6:]  # Remove "slack/" prefix
                display_name = f"Slack: #{channel}"
            else:
                display_name = f"Slack: {source}"
            
            return DataSourceInfo(
                display_name=display_name,
                source_type="slack",
                original_source=source,
                metadata={"channel": channel if 'channel' in locals() else source}
            )
        except Exception as e:
            logger.warning(f"Error parsing Slack source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_teams_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format Microsoft Teams channel references."""
        try:
            # Teams format: "team-name/channel-name"
            if "/" in source:
                team, channel = source.split("/", 1)
                display_name = f"Teams: {team} - {channel}"
            else:
                display_name = f"Teams: {source}"
            
            return DataSourceInfo(
                display_name=display_name,
                source_type="teams",
                original_source=source,
                metadata={"team": team if 'team' in locals() else None, "channel": channel if 'channel' in locals() else source}
            )
        except Exception as e:
            logger.warning(f"Error parsing Teams source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_outlook_email_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format Outlook email URLs."""
        try:
            parsed = urlparse(source)
            domain = parsed.netloc
            
            # Extract message ID from path
            path_parts = parsed.path.strip("/").split("/")
            if "mail" in path_parts and "id" in path_parts:
                id_index = path_parts.index("id")
                if id_index + 1 < len(path_parts):
                    message_id = path_parts[id_index + 1]
                    display_name = f"Outlook Email: {message_id[:8]}..."
                else:
                    display_name = "Outlook Email"
            else:
                display_name = "Outlook Email"
            
            return DataSourceInfo(
                display_name=display_name,
                source_type="outlook_email",
                original_source=source,
                metadata={"domain": domain, "message_id": message_id if 'message_id' in locals() else None}
            )
        except Exception as e:
            logger.warning(f"Error parsing Outlook email source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_servicenow_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format ServiceNow URLs."""
        try:
            parsed = urlparse(source)
            domain = parsed.netloc
            
            # Extract sys_id from query parameters
            query_params = parsed.query
            if "sys_id=" in query_params:
                sys_id_start = query_params.find("sys_id=") + 7
                sys_id_end = query_params.find("&", sys_id_start)
                if sys_id_end == -1:
                    sys_id = query_params[sys_id_start:]
                else:
                    sys_id = query_params[sys_id_start:sys_id_end]
                display_name = f"ServiceNow: {sys_id[:8]}..."
            else:
                display_name = "ServiceNow"
            
            return DataSourceInfo(
                display_name=display_name,
                source_type="servicenow",
                original_source=source,
                metadata={"domain": domain, "sys_id": sys_id if 'sys_id' in locals() else None}
            )
        except Exception as e:
            logger.warning(f"Error parsing ServiceNow source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_meeting_notes_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format Google Cloud Storage meeting notes URLs."""
        try:
            # GCS format: "gs://bucket-name/path/to/file"
            if source.startswith("gs://"):
                path_without_prefix = source[5:]  # Remove "gs://" prefix
                path_parts = path_without_prefix.split("/")
                
                if len(path_parts) >= 2:
                    bucket = path_parts[0]
                    file_path = "/".join(path_parts[1:])
                    file_name = path_parts[-1] if path_parts else "unknown"
                    display_name = f"Meeting Notes: {file_name}"
                else:
                    display_name = "Meeting Notes"
            else:
                display_name = "Meeting Notes"
            
            return DataSourceInfo(
                display_name=display_name,
                source_type="meeting_notes",
                original_source=source,
                metadata={"bucket": bucket if 'bucket' in locals() else None, "file_path": file_path if 'file_path' in locals() else None}
            )
        except Exception as e:
            logger.warning(f"Error parsing meeting notes source '{source}': {e}")
            return self._format_default_source(source, context)
    
    def _format_calendar_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Format calendar/meeting sources."""
        return DataSourceInfo(
            display_name="Outlook Calendar",
            source_type="calendar",
            original_source=source,
            metadata={"calendar_type": "outlook"}
        )
    
    def _format_default_source(self, source: str, context: Dict[str, Any] = None) -> DataSourceInfo:
        """Default formatter for unknown source types."""
        # Try to extract meaningful information
        if source.startswith("http"):
            parsed = urlparse(source)
            domain = parsed.netloc
            display_name = f"Knowledge base: {domain}"
        else:
            # Clean up the source name
            clean_name = source.replace("_", " ").replace("-", " ").title()
            display_name = f"Knowledge base: {clean_name}"
        
        return DataSourceInfo(
            display_name=display_name,
            source_type="default",
            original_source=source,
            metadata={"fallback": True}
        )
    
    def add_custom_formatter(self, source_type: str, formatter_func, detection_pattern: str = None):
        """
        Add a custom formatter for a new source type.
        
        Args:
            source_type: The type identifier for the source
            formatter_func: Function that takes (source, context) and returns DataSourceInfo
            detection_pattern: Optional regex pattern for automatic detection
        """
        self.formatters[source_type] = formatter_func
        if detection_pattern:
            self.detection_patterns[source_type] = detection_pattern
        logger.info(f"Added custom formatter for source type: {source_type}")
    
    def get_source_statistics(self, sources: List[str]) -> Dict[str, Any]:
        """
        Get statistics about the data sources.
        
        Args:
            sources: List of source strings
            
        Returns:
            Dictionary with source statistics
        """
        stats = {
            "total_sources": len(sources),
            "source_types": {},
            "domains": set(),
            "unique_sources": len(set(sources))
        }
        
        for source in sources:
            source_type = self._detect_source_type(source.lower())
            stats["source_types"][source_type] = stats["source_types"].get(source_type, 0) + 1
            
            if source.startswith("http"):
                try:
                    parsed = urlparse(source)
                    stats["domains"].add(parsed.netloc)
                except:
                    pass
        
        stats["domains"] = list(stats["domains"])
        return stats
