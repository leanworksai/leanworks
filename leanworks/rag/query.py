from typing import List
import json
import re
from functools import lru_cache

class QueryParser:
    """
    Class for parsing user queries and generating clarification questions
    when queries are vague or lack context.
    """
    
    def __init__(self, model_client=None):
        """
        Initialize the QueryParser with a model client.
        
        Args:
            model_client: Initialized client for LLM generation (e.g., OpenAI client)
        """
        self.model_client = model_client
    
    def parse_query(self, query: str, max_num_queries: int = 2) -> List[str]:
        """
        Analyze the user query and generate context gathering statements to help retrieve more relevant information.
        
        Args:
            query: The original user query
            max_questions: Maximum number of context gathering statements to generate
            
        Returns:
            List of statements indicating what additional context would be helpful
        """
        prompt = f"Original query: {query}\n\nGenerate {max_num_queries} alternative search queries that would help retrieve relevant information for this query. Focus on different aspects or facets of the original query. Return each query on a new line."
    
        
        response = self.model_client.chat.completions.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that helps create alternative search queries for information retrieval."},
                {"role": "user", "content": prompt}
            ]
        )
        
        parsed_queries = []
        answer = response.choices[0].message.content
        
        # Extract context needs from response
        try:
            context_needs = json.loads(self.extract_first_brace_or_bracket_content(answer))
        except (json.JSONDecodeError, AttributeError):
            # Fallback if JSON parsing fails
            context_needs = [f"Additional context needed regarding: {query}"]
            
        # Limit to requested number
        return context_needs[:max_num_queries]

    def extract_first_brace_or_bracket_content(self, string):
        """
        Extract the first JSON-like content (enclosed by {} or []) from a string.
        
        Args:
            string: The input string to extract JSON content from
            
        Returns:
            Properly formatted JSON string or error message
        """
        # Define a regex pattern to match anything enclosed by {} or []
        pattern = r'(\{[^{}]*\}|\[[^\[\]]*\])'
        
        # Find the first occurrence of either pattern
        match = re.search(pattern, string)
        
        if match:
            content = match.group(0)
            try:
                # Try to parse as JSON to validate and fix any escape issues
                parsed = json.loads(content)
                return json.dumps(parsed)  # Return properly formatted JSON string
            except json.JSONDecodeError as e:
                # If JSON parsing fails, try to clean up common escape sequence issues
                cleaned = content.replace('\\', '\\\\')  # Double up backslashes
                try:
                    parsed = json.loads(cleaned)
                    return json.dumps(parsed)
                except json.JSONDecodeError:
                    # If still fails, return original content for backward compatibility
                    return content
        
        return "No JSON data available in response."