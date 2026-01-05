"""
Caching utilities for API keys and client instances
"""
import time
from typing import Dict, Tuple, Optional

# Simplified caching: Only cache expensive operations (API keys and client instances)
# Using a simple dict with TTL - sufficient for this use case
_cache: Dict[str, Tuple[any, float]] = {}
_cache_ttl = 300  # 5 minutes

def get_cache(key: str) -> Optional[any]:
    """Get from cache if not expired"""
    if key in _cache:
        value, timestamp = _cache[key]
        if time.time() - timestamp < _cache_ttl:
            return value
        del _cache[key]
    return None

def set_cache(key: str, value: any):
    """Set cache value with timestamp"""
    _cache[key] = (value, time.time())

def clear_cache():
    """Clear all caches"""
    _cache.clear()

