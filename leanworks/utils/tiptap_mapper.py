"""
TipTap position mapping utilities.

Maps ProseMirror positions (from TipTap JSON) to HTML character positions.
This allows the agent to work with HTML positions for document editing while
the frontend uses ProseMirror positions.
"""
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def parse_tiptap_json(content: Union[str, dict]) -> Optional[dict]:
    """
    Parse TipTap JSON content.
    
    Args:
        content: TipTap JSON as string or dict
        
    Returns:
        Parsed TipTap JSON dict or None if invalid
    """
    if not content:
        return None
    
    if isinstance(content, dict):
        if content.get("type") == "doc":
            return content
        return None
    
    if isinstance(content, str):
        try:
            doc = json.loads(content)
            if isinstance(doc, dict) and doc.get("type") == "doc":
                return doc
        except json.JSONDecodeError:
            pass
    
    return None


def calculate_node_size(node: dict) -> int:
    """
    Calculate the size of a TipTap node in ProseMirror positions.
    
    Args:
        node: TipTap node dict
        
    Returns:
        Size in ProseMirror positions
    """
    if node.get("type") == "text":
        return len(node.get("text", ""))
    
    # Non-text nodes contribute 1 for opening, 1 for closing
    size = 2
    
    if node.get("content"):
        for child in node["content"]:
            size += calculate_node_size(child)
    
    return size


def map_prosemirror_to_html_positions(
    doc_json: dict,
    from_pos: int,
    to_pos: int
) -> Tuple[int, int]:
    """
    Map ProseMirror positions to HTML character positions.
    
    This function traverses the TipTap JSON structure and converts it to HTML
    while tracking position mappings.
    
    Args:
        doc_json: TipTap JSON document (must have type="doc")
        from_pos: ProseMirror start position
        to_pos: ProseMirror end position
        
    Returns:
        Tuple of (html_start, html_end) character positions
    """
    if not doc_json or doc_json.get("type") != "doc":
        logger.warning("Invalid TipTap JSON document")
        return (0, 0)
    
    html_parts = []
    position_map = []  # List of (prosemirror_pos, html_pos) tuples
    
    def process_list_item(node: dict, prosemirror_offset: int, html_offset: int) -> Tuple[int, int]:
        """Process a list item node."""
        html_start_tag = "<li>"
        html_parts.append(html_start_tag)
        html_offset += len(html_start_tag)
        prosemirror_offset += 1
        
        for child in node.get("content", []):
            prosemirror_offset, html_offset = process_node(child, prosemirror_offset, html_offset)
        
        html_end_tag = "</li>"
        html_parts.append(html_end_tag)
        html_offset += len(html_end_tag)
        prosemirror_offset += 1
        
        return (prosemirror_offset, html_offset)
    
    def process_node(node: dict, prosemirror_offset: int, html_offset: int) -> Tuple[int, int]:
        """
        Process a node and return (prosemirror_end, html_end).
        
        Args:
            node: TipTap node
            prosemirror_offset: Current ProseMirror position
            html_offset: Current HTML position
            
        Returns:
            Tuple of (prosemirror_end, html_end)
        """
        node_type = node.get("type", "")
        attrs = node.get("attrs", {})
        node_content = node.get("content", [])
        
        if node_type == "text":
            text = node.get("text", "")
            marks = node.get("marks", [])
            
            # Track position at start of text
            position_map.append((prosemirror_offset, html_offset))
            
            # Apply marks (bold, italic, etc.) as HTML tags
            html_text = _escape_html(text)
            for mark in reversed(marks):  # Reverse to apply inner marks first
                mark_type = mark.get("type", "")
                if mark_type == "bold":
                    html_text = f"<strong>{html_text}</strong>"
                elif mark_type == "italic":
                    html_text = f"<em>{html_text}</em>"
                elif mark_type == "strike":
                    html_text = f"<s>{html_text}</s>"
                elif mark_type == "code":
                    html_text = f"<code>{html_text}</code>"
                elif mark_type == "link":
                    href = mark.get("attrs", {}).get("href", "")
                    href_escaped = _escape_html(href)
                    html_text = f'<a href="{href_escaped}" class="text-primary underline">{html_text}</a>'
            
            html_parts.append(html_text)
            prosemirror_end = prosemirror_offset + len(text)
            html_end = html_offset + len(html_text)
            
            # Track position at end of text
            position_map.append((prosemirror_end, html_end))
            
            return (prosemirror_end, html_end)
        
        elif node_type == "paragraph":
            html_start_tag = "<p>"
            html_parts.append(html_start_tag)
            html_offset += len(html_start_tag)
            
            prosemirror_offset += 1  # Opening tag
            
            # Process children
            for child in node_content:
                prosemirror_offset, html_offset = process_node(child, prosemirror_offset, html_offset)
            
            html_end_tag = "</p>"
            html_parts.append(html_end_tag)
            html_offset += len(html_end_tag)
            prosemirror_offset += 1  # Closing tag
            
            return (prosemirror_offset, html_offset)
        
        elif node_type == "heading":
            level = attrs.get("level", 1)
            html_start_tag = f"<h{level}>"
            html_parts.append(html_start_tag)
            html_offset += len(html_start_tag)
            
            prosemirror_offset += 1  # Opening tag
            
            # Process children
            for child in node_content:
                prosemirror_offset, html_offset = process_node(child, prosemirror_offset, html_offset)
            
            html_end_tag = f"</h{level}>"
            html_parts.append(html_end_tag)
            html_offset += len(html_end_tag)
            prosemirror_offset += 1  # Closing tag
            
            return (prosemirror_offset, html_offset)
        
        elif node_type == "bulletList":
            html_start_tag = "<ul>"
            html_parts.append(html_start_tag)
            html_offset += len(html_start_tag)
            prosemirror_offset += 1
            
            for child in node_content:
                if child.get("type") == "listItem":
                    prosemirror_offset, html_offset = process_list_item(child, prosemirror_offset, html_offset)
            
            html_end_tag = "</ul>"
            html_parts.append(html_end_tag)
            html_offset += len(html_end_tag)
            prosemirror_offset += 1
            
            return (prosemirror_offset, html_offset)
        
        elif node_type == "orderedList":
            html_start_tag = "<ol>"
            html_parts.append(html_start_tag)
            html_offset += len(html_start_tag)
            prosemirror_offset += 1
            
            for child in node_content:
                if child.get("type") == "listItem":
                    prosemirror_offset, html_offset = process_list_item(child, prosemirror_offset, html_offset)
            
            html_end_tag = "</ol>"
            html_parts.append(html_end_tag)
            html_offset += len(html_end_tag)
            prosemirror_offset += 1
            
            return (prosemirror_offset, html_offset)
        
        elif node_type == "listItem":
            return process_list_item(node, prosemirror_offset, html_offset)
        
        elif node_type == "blockquote":
            html_start_tag = "<blockquote>"
            html_parts.append(html_start_tag)
            html_offset += len(html_start_tag)
            prosemirror_offset += 1
            
            for child in node_content:
                prosemirror_offset, html_offset = process_node(child, prosemirror_offset, html_offset)
            
            html_end_tag = "</blockquote>"
            html_parts.append(html_end_tag)
            html_offset += len(html_end_tag)
            prosemirror_offset += 1
            
            return (prosemirror_offset, html_offset)
        
        elif node_type == "codeBlock":
            html_start_tag = "<pre><code>"
            html_parts.append(html_start_tag)
            html_offset += len(html_start_tag)
            prosemirror_offset += 1
            
            for child in node_content:
                prosemirror_offset, html_offset = process_node(child, prosemirror_offset, html_offset)
            
            html_end_tag = "</code></pre>"
            html_parts.append(html_end_tag)
            html_offset += len(html_end_tag)
            prosemirror_offset += 1
            
            return (prosemirror_offset, html_offset)
        
        elif node_type == "hardBreak":
            html_parts.append("<br/>")
            html_offset += 5
            prosemirror_offset += 1
            return (prosemirror_offset, html_offset)
        
        else:
            # Unknown node type - process children if any
            prosemirror_offset += 1  # Opening
            for child in node_content:
                prosemirror_offset, html_offset = process_node(child, prosemirror_offset, html_offset)
            prosemirror_offset += 1  # Closing
            return (prosemirror_offset, html_offset)
    
    # Process document starting from position 1 (after doc opening)
    prosemirror_pos = 1
    html_pos = 0
    
    content = doc_json.get("content", [])
    for node in content:
        prosemirror_pos, html_pos = process_node(node, prosemirror_pos, html_pos)
    
    # Build HTML string
    html_string = "".join(html_parts)
    
    # Find HTML positions for given ProseMirror positions
    html_from = _find_html_position(position_map, from_pos)
    html_to = _find_html_position(position_map, to_pos)
    
    return (html_from, html_to)


def _find_html_position(position_map: List[Tuple[int, int]], prosemirror_pos: int) -> int:
    """
    Find HTML position for a given ProseMirror position using position map.
    
    Args:
        position_map: List of (prosemirror_pos, html_pos) tuples
        prosemirror_pos: ProseMirror position to find
        
    Returns:
        HTML position
    """
    if not position_map:
        return 0
    
    # Find closest mapping
    for i, (pm_pos, html_pos) in enumerate(position_map):
        if pm_pos >= prosemirror_pos:
            if i > 0:
                # Interpolate between previous and current
                prev_pm, prev_html = position_map[i - 1]
                if pm_pos > prev_pm:
                    ratio = (prosemirror_pos - prev_pm) / (pm_pos - prev_pm)
                    return int(prev_html + ratio * (html_pos - prev_html))
            return html_pos
    
    # If beyond all mappings, use last HTML position
    return position_map[-1][1]


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def get_block_node_at_position(doc_json: dict, block_pos: int) -> Optional[dict]:
    """
    Get block node at a specific ProseMirror position.
    
    Args:
        doc_json: TipTap JSON document
        block_pos: Block start position in ProseMirror document
        
    Returns:
        Block node dict or None if not found
    """
    if not doc_json or doc_json.get("type") != "doc":
        return None
    
    content = doc_json.get("content", [])
    current_pos = 1  # Start after doc opening
    
    def find_block(nodes: List[dict], pos: int) -> Optional[dict]:
        for node in nodes:
            node_size = calculate_node_size(node)
            node_start = pos
            node_end = pos + node_size
            
            if block_pos >= node_start and block_pos < node_end:
                node_type = node.get("type", "")
                if node_type in ["paragraph", "heading", "listItem", "blockquote", "codeBlock", "tableCell", "tableHeader"]:
                    return node
                
                if node.get("content"):
                    result = find_block(node["content"], pos + 1)
                    if result:
                        return result
            
            pos = node_end
        
        return None
    
    return find_block(content, current_pos)


def convert_selected_text_positions(
    selected_text_obj: dict,
    doc_content: Union[str, dict]
) -> dict:
    """
    Convert selected text positions from ProseMirror to HTML.
    
    Main conversion function that takes selected text object and document content,
    and returns updated object with HTML positions.
    
    Args:
        selected_text_obj: Selected text object with ProseMirror positions
            {
                "text": "selected text",
                "docId": "doc_123",
                "from": 100,
                "to": 150,
                "blockType": "paragraph",
                "blockPos": 95,
                "blockOffset": 5
            }
        doc_content: Document content (TipTap JSON string or dict)
        
    Returns:
        Dictionary with HTML positions added:
        {
            "htmlFrom": 120,
            "htmlTo": 180,
            "htmlBlockPos": 115,
            ... (original fields preserved)
        }
    """
    # Parse document content
    doc_json = parse_tiptap_json(doc_content)
    if not doc_json:
        logger.warning("Failed to parse document content for position conversion")
        return selected_text_obj.copy()
    
    result = selected_text_obj.copy()
    
    # Convert main positions (from, to)
    from_pos = selected_text_obj.get("from")
    to_pos = selected_text_obj.get("to")
    
    if from_pos is not None and to_pos is not None:
        try:
            html_from, html_to = map_prosemirror_to_html_positions(doc_json, from_pos, to_pos)
            result["htmlFrom"] = html_from
            result["htmlTo"] = html_to
        except Exception as e:
            logger.error(f"Error converting positions: {str(e)}")
    
    # Convert block position if available
    block_pos = selected_text_obj.get("blockPos")
    if block_pos is not None:
        try:
            # For block position, we need to find where the block starts in HTML
            # This is approximate - we find the HTML position at block_pos
            html_block_from, _ = map_prosemirror_to_html_positions(doc_json, block_pos, block_pos)
            result["htmlBlockPos"] = html_block_from
        except Exception as e:
            logger.error(f"Error converting block position: {str(e)}")
    
    return result
