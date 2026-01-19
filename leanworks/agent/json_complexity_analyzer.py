from typing import Any, Dict, Tuple, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """JSON complexity classification levels"""
    SIMPLE = "simple"        # Flat/Tabular: depth ≤ 3, consistent schema
    COMPLEX = "complex"      # Deep nesting: depth > 3, complex hierarchies
    NOT_JSON = "not_json"    # Not JSON data


class JSONComplexityAnalyzer:
    """
    Analyzes JSON structures to determine complexity level for optimal storage routing.

    Simple JSON → DuckDB (tabular queries)
    Complex JSON → jq (navigation/transformation)
    """

    @classmethod
    def analyze(cls, data: Any) -> Dict[str, Any]:
        """
        Analyze JSON data structure and return complexity information.

        Args:
            data: The data to analyze (dict, list, or other)

        Returns:
            Dict with complexity analysis:
            {
                "level": ComplexityLevel,
                "max_depth": int,
                "key_count": int,
                "has_nested_arrays": bool,
                "has_mixed_types": bool,
                "estimated_complexity": str
            }
        """
        if not isinstance(data, (dict, list)):
            return {
                "level": ComplexityLevel.NOT_JSON,
                "max_depth": 0,
                "key_count": 0,
                "has_nested_arrays": False,
                "has_mixed_types": False,
                "estimated_complexity": "not_json"
            }

        # Analyze the structure
        analysis = cls._analyze_structure(data)

        # Determine complexity level
        level = cls._determine_complexity_level(analysis)

        return {
            "level": level,
            "max_depth": analysis["max_depth"],
            "key_count": analysis["key_count"],
            "has_nested_arrays": analysis["has_nested_arrays"],
            "has_mixed_types": analysis["has_mixed_types"],
            "estimated_complexity": level.value
        }

    @classmethod
    def _analyze_structure(cls, data: Union[Dict, List], current_depth: int = 0) -> Dict[str, Any]:
        """
        Recursively analyze the structure of JSON data.

        Args:
            data: Current data node
            current_depth: Current nesting depth

        Returns:
            Analysis dictionary
        """
        max_depth = current_depth
        key_count = 0
        has_nested_arrays = False
        has_mixed_types = False

        if isinstance(data, dict):
            key_count = len(data.keys())

            for key, value in data.items():
                # Recursively analyze nested structures
                nested_analysis = cls._analyze_structure(value, current_depth + 1)
                max_depth = max(max_depth, nested_analysis["max_depth"])
                has_nested_arrays = has_nested_arrays or nested_analysis["has_nested_arrays"]
                has_mixed_types = has_mixed_types or nested_analysis["has_mixed_types"]

        elif isinstance(data, list):
            if not data:  # Empty list
                return {
                    "max_depth": current_depth,
                    "key_count": 0,
                    "has_nested_arrays": False,
                    "has_mixed_types": False
                }

            # Check first few items for type consistency
            sample_size = min(5, len(data))
            types_seen = set()

            for i, item in enumerate(data[:sample_size]):
                item_type = type(item).__name__
                types_seen.add(item_type)

                # Analyze nested structure
                nested_analysis = cls._analyze_structure(item, current_depth + 1)
                max_depth = max(max_depth, nested_analysis["max_depth"])
                has_nested_arrays = has_nested_arrays or nested_analysis["has_nested_arrays"]

                # Check for complex objects in arrays
                if isinstance(item, (dict, list)) and current_depth >= 1:
                    has_nested_arrays = True

            # Check for mixed types in array
            has_mixed_types = len(types_seen) > 1 and any(t in ['dict', 'list'] for t in types_seen)

        return {
            "max_depth": max_depth,
            "key_count": key_count,
            "has_nested_arrays": has_nested_arrays,
            "has_mixed_types": has_mixed_types
        }

    @classmethod
    def _determine_complexity_level(cls, analysis: Dict[str, Any]) -> ComplexityLevel:
        """
        Determine complexity level based on structural analysis.

        Args:
            analysis: Structure analysis from _analyze_structure()

        Returns:
            ComplexityLevel enum value
        """
        max_depth = analysis["max_depth"]
        has_nested_arrays = analysis["has_nested_arrays"]
        has_mixed_types = analysis["has_mixed_types"]

        # SIMPLE criteria (DuckDB suitable):
        # - Depth ≤ 3
        # - No nested arrays of complex objects
        # - No mixed types in arrays
        if max_depth <= 3 and not has_nested_arrays and not has_mixed_types:
            return ComplexityLevel.SIMPLE

        # COMPLEX criteria (jq suitable):
        # - Depth > 3, OR
        # - Has nested arrays of complex objects, OR
        # - Has mixed types in arrays
        else:
            return ComplexityLevel.COMPLEX

    @classmethod
    def is_simple_json(cls, data: Any) -> bool:
        """
        Convenience method to check if data is simple JSON suitable for DuckDB.

        Args:
            data: Data to check

        Returns:
            True if simple JSON, False otherwise
        """
        analysis = cls.analyze(data)
        return analysis["level"] == ComplexityLevel.SIMPLE

    @classmethod
    def is_complex_json(cls, data: Any) -> bool:
        """
        Convenience method to check if data is complex JSON suitable for jq.

        Args:
            data: Data to check

        Returns:
            True if complex JSON, False otherwise
        """
        analysis = cls.analyze(data)
        return analysis["level"] == ComplexityLevel.COMPLEX

    @classmethod
    def get_complexity_description(cls, data: Any) -> str:
        """
        Get human-readable description of JSON complexity.

        Args:
            data: Data to analyze

        Returns:
            Description string
        """
        analysis = cls.analyze(data)

        if analysis["level"] == ComplexityLevel.NOT_JSON:
            return "Not JSON data"

        elif analysis["level"] == ComplexityLevel.SIMPLE:
            return f"Simple JSON (depth: {analysis['max_depth']}, keys: {analysis['key_count']})"

        else:  # COMPLEX
            features = []
            if analysis["max_depth"] > 3:
                features.append(f"deep nesting ({analysis['max_depth']} levels)")
            if analysis["has_nested_arrays"]:
                features.append("nested arrays")
            if analysis["has_mixed_types"]:
                features.append("mixed types")

            feature_str = ", ".join(features) if features else "complex structure"
            return f"Complex JSON ({feature_str})"