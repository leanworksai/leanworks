from typing import Dict, Optional, List
from datetime import datetime, timezone
import json
import re
from leanworks.rag.setting import OTHER_MODEL

class FilterExtractor:
    """
    Class for extracting time filters from user queries using Anthropic.
    """
    
    def __init__(self):
        pass
    
    def extract_time_filters(self, query: str, model_client) -> Dict[str, Optional[int]]:
        """
        Extract time filter (start_date and end_date) from user query using Anthropic.
        
        Args:
            query: The user query
            model_client: Initialized model client for LLM generation (can be OpenAI or Anthropic)
        Returns:
            Dictionary with 'start_timestamp' and 'end_timestamp' as keys and Unix timestamps as values or None if not found
        """
        current_timestamp = int(datetime.now(timezone.utc).timestamp())
        today_date = datetime.now(timezone.utc).isoformat() + "Z"
        
        prompt = f"""
        Extract any time filters or date ranges from the following query. 
        If there are specific dates mentioned, convert them to ISO format (YYYY-MM-DDTHH:MM:SSZ) first.
        If there are relative time references (like "last week", "past month", etc.), convert them to actual dates.
        If the query contains words like "recent" or "latest" without any specific time frame, interpret this as looking back 1 week from today.
        
        Today's date is {today_date}.
        All dates should be interpreted in UTC timezone.
        
        Query: {query}
        
        Return your answer in JSON format with the following structure:
        {{
            "start_datetime": "YYYY-MM-DDTHH:MM:SSZ or null if not specified",
            "end_datetime": "YYYY-MM-DDTHH:MM:SSZ or null if not specified"
        }}
        """
        
        try:
            response = model_client.chat.completions.create(
                model=OTHER_MODEL,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts time filters from queries. Always interpret dates in UTC timezone."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = json.loads(response.choices[0].message.content)
            start_datetime = result.get("start_datetime")
            end_datetime = result.get("end_datetime")
            # Convert ISO dates to Unix timestamps (in UTC)
            start_timestamp = int(datetime.fromisoformat(start_datetime.replace("Z", "+00:00")).timestamp()) if start_datetime else None
            end_timestamp = int(datetime.fromisoformat(end_datetime.replace("Z", "+00:00")).timestamp()) if end_datetime else None
                    
            start_diff = abs(start_timestamp - current_timestamp) if start_timestamp else 0
            end_diff = abs(end_timestamp - current_timestamp) if end_timestamp else 0
            
            # Check if start_date or end_date is in the future
            if (start_timestamp and start_timestamp > current_timestamp) or (end_timestamp and end_timestamp > current_timestamp):

                # Determine which date is farthest from current date
                if start_diff > end_diff:
                    return {
                        "start_timestamp": int(current_timestamp - start_diff),
                        "end_timestamp": int(current_timestamp)
                    }
                else:
                    return {
                        "start_timestamp": int(current_timestamp - end_diff),
                        "end_timestamp": int(current_timestamp)
                    }
            else:
                return {
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp
                }
        except Exception as e:
            print(f"Error extracting time filters: {e}")
            # Return None values if parsing fails
            return {
                "start_timestamp": None,
                "end_timestamp": None
            }
    def extract_user_filters(self, query: str) -> List[str]:
        """
        Extract user email addresses from the query using regex.
        
        Args:
            query: The user query
            
        Returns:
            List of unique user email addresses found in the query
        """
        # Regex pattern for matching email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        # Find all email matches in the query
        email_matches = re.findall(email_pattern, query)
        
        # Return unique email addresses (remove duplicates)
        return list(set(email_matches))