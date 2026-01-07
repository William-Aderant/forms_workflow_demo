"""
Field Candidate Detection Service
=================================

Identifies empty fillable fields from Textract output using:
1. Horizontal lines with no overlapping text (line fields)
2. Empty table cells
3. Checkboxes (SELECTION_ELEMENT blocks) without filled selections

Design Principles:
------------------
- Geometry is ALWAYS from Textract (never modified)
- Conservative detection: prefer false negatives over false positives
- All candidates include nearby context for semantic classification

Tradeoffs:
----------
1. Line detection uses geometry heuristics:
   - Aspect ratio > 8:1 suggests horizontal line
   - This may miss very short fields (3-4 char width)
   - Mitigation: also detect by explicit "____" patterns in text

2. Checkbox detection relies on SELECTION_ELEMENT:
   - Textract may miss hand-drawn checkboxes
   - Mitigation: can be enhanced with custom CV detection

3. Table cell detection uses TABLE/CELL blocks:
   - Empty cells identified by lack of child WORD blocks
   - Nested tables may cause duplicate detection
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set
import math

logger = logging.getLogger(__name__)


class FieldType(str, Enum):
    """Types of fillable fields detected on forms."""
    LINE = "line"           # Horizontal line for text entry
    CHECKBOX = "checkbox"   # Checkbox or selection element
    TABLE_CELL = "table_cell"  # Empty cell in a table


@dataclass
class BoundingBox:
    """
    Normalized bounding box coordinates.
    
    All values are normalized to [0, 1] range relative to page dimensions.
    This matches Textract's coordinate system exactly.
    """
    left: float    # X coordinate of left edge
    top: float     # Y coordinate of top edge  
    width: float   # Width of bounding box
    height: float  # Height of bounding box
    
    @property
    def right(self) -> float:
        """X coordinate of right edge."""
        return self.left + self.width
    
    @property
    def bottom(self) -> float:
        """Y coordinate of bottom edge."""
        return self.top + self.height
    
    @property
    def center_x(self) -> float:
        """X coordinate of center."""
        return self.left + self.width / 2
    
    @property
    def center_y(self) -> float:
        """Y coordinate of center."""
        return self.top + self.height / 2
    
    @property
    def area(self) -> float:
        """Area of bounding box."""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """Width to height ratio (>1 means wider than tall)."""
        if self.height == 0:
            return float('inf')
        return self.width / self.height
    
    def to_list(self) -> List[float]:
        """Convert to [x1, y1, x2, y2] format."""
        return [self.left, self.top, self.right, self.bottom]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format matching Textract."""
        return {
            'left': self.left,
            'top': self.top,
            'width': self.width,
            'height': self.height
        }
    
    @classmethod
    def from_textract(cls, bbox_dict: Dict[str, float]) -> 'BoundingBox':
        """Create from Textract bounding box dictionary."""
        return cls(
            left=bbox_dict.get('Left', bbox_dict.get('left', 0)),
            top=bbox_dict.get('Top', bbox_dict.get('top', 0)),
            width=bbox_dict.get('Width', bbox_dict.get('width', 0)),
            height=bbox_dict.get('Height', bbox_dict.get('height', 0))
        )
    
    def overlaps(self, other: 'BoundingBox', threshold: float = 0.0) -> bool:
        """
        Check if this box overlaps with another.
        
        Args:
            other: Another bounding box
            threshold: Minimum overlap ratio to consider as overlap
        
        Returns:
            True if boxes overlap by more than threshold
        """
        # Calculate intersection
        x_overlap = max(0, min(self.right, other.right) - max(self.left, other.left))
        y_overlap = max(0, min(self.bottom, other.bottom) - max(self.top, other.top))
        intersection = x_overlap * y_overlap
        
        if threshold == 0:
            return intersection > 0
        
        # Calculate overlap ratio relative to smaller box
        min_area = min(self.area, other.area)
        if min_area == 0:
            return False
        return intersection / min_area > threshold
    
    def distance_to(self, other: 'BoundingBox') -> float:
        """Calculate center-to-center distance."""
        dx = self.center_x - other.center_x
        dy = self.center_y - other.center_y
        return math.sqrt(dx * dx + dy * dy)
    
    def is_left_of(self, other: 'BoundingBox', tolerance: float = 0.02) -> bool:
        """Check if this box is to the left of another."""
        return self.right <= other.left + tolerance
    
    def is_above(self, other: 'BoundingBox', tolerance: float = 0.02) -> bool:
        """Check if this box is above another."""
        return self.bottom <= other.top + tolerance
    
    def same_row(self, other: 'BoundingBox', tolerance: float = 0.03) -> bool:
        """Check if two boxes are on the same horizontal row."""
        return abs(self.center_y - other.center_y) < tolerance


@dataclass
class TextBlock:
    """A text element from Textract with its spatial information."""
    text: str
    bounding_box: BoundingBox
    block_type: str  # WORD, LINE, etc.
    confidence: float
    block_id: str


@dataclass
class FieldCandidate:
    """
    A detected empty fillable field candidate.
    
    This represents a location on the form where user input is expected
    but currently empty.
    """
    candidate_id: str
    field_type: FieldType
    bounding_box: BoundingBox
    page_number: int
    
    # Nearby text context for semantic classification
    nearby_text: List[TextBlock] = field(default_factory=list)
    
    # Supporting text (most likely label)
    supporting_text: str = ""
    
    # Detection metadata
    detection_method: str = ""  # e.g., "geometry_line", "selection_element", "empty_cell"
    detection_confidence: float = 1.0
    
    # For checkboxes: selection state from Textract
    selection_status: Optional[str] = None  # "SELECTED" or "NOT_SELECTED"
    
    # For table cells: table context
    table_id: Optional[str] = None
    row_index: Optional[int] = None
    col_index: Optional[int] = None
    
    def get_context_text(self, max_chars: int = 200) -> str:
        """
        Get concatenated nearby text for context.
        
        Args:
            max_chars: Maximum characters to return
        
        Returns:
            Concatenated nearby text
        """
        texts = [tb.text for tb in self.nearby_text]
        combined = " ".join(texts)
        if len(combined) > max_chars:
            return combined[:max_chars] + "..."
        return combined


class FieldCandidateDetector:
    """
    Detects empty fillable field candidates from Textract output.
    
    Detection strategies:
    1. SELECTION_ELEMENT blocks → checkboxes
    2. Geometry-based line detection → horizontal fill lines
    3. TABLE/CELL analysis → empty table cells
    4. Underline pattern detection → "____" text patterns
    5. KEY_VALUE_SET blocks → Textract-detected form fields (VALUE areas)
    """
    
    # Geometry thresholds for line detection
    # Tradeoff: Higher threshold = fewer false positives, may miss short fields
    # Relaxed from 6.0/0.05 to catch more fields including shorter ones
    MIN_LINE_ASPECT_RATIO = 4.0      # Width must be 4x height (was 6.0)
    MIN_LINE_WIDTH = 0.03            # Minimum 3% of page width (was 0.05)
    MAX_LINE_HEIGHT = 0.025          # Maximum 2.5% of page height (was 0.02)
    
    # Context collection radius (normalized coordinates)
    # Expanded to catch labels that are further from fields
    CONTEXT_RADIUS_X = 0.20          # 20% of page width (was 0.15)
    CONTEXT_RADIUS_Y = 0.10          # 10% of page height (was 0.08)
    
    # Underline pattern for text-based detection
    UNDERLINE_PATTERN_MIN_LENGTH = 3  # Minimum "_" characters
    
    def __init__(self):
        """Initialize the detector."""
        self._candidate_counter = 0
    
    def detect_candidates(
        self,
        textract_blocks: List[Dict[str, Any]],
        page_number: int = 1
    ) -> List[FieldCandidate]:
        """
        Detect all empty field candidates from Textract blocks.
        
        Args:
            textract_blocks: Raw Textract block list
            page_number: Page number (1-indexed)
        
        Returns:
            List of FieldCandidate objects
        """
        logger.info(f"Detecting field candidates on page {page_number}")
        
        # Build block index for efficient lookup
        block_map = {block['Id']: block for block in textract_blocks if 'Id' in block}
        
        # Collect all text blocks for context
        text_blocks = self._extract_text_blocks(textract_blocks)
        
        candidates = []
        
        # Strategy 1: Detect checkboxes from SELECTION_ELEMENT
        checkbox_candidates = self._detect_checkboxes(textract_blocks, block_map, page_number)
        candidates.extend(checkbox_candidates)
        logger.info(f"Detected {len(checkbox_candidates)} checkbox candidates")
        
        # Strategy 2: Detect empty table cells
        table_cell_candidates = self._detect_empty_table_cells(textract_blocks, block_map, page_number)
        candidates.extend(table_cell_candidates)
        logger.info(f"Detected {len(table_cell_candidates)} empty table cell candidates")
        
        # Strategy 3: Detect horizontal lines (geometry-based)
        line_candidates = self._detect_geometry_lines(textract_blocks, text_blocks, page_number)
        candidates.extend(line_candidates)
        logger.info(f"Detected {len(line_candidates)} line field candidates")
        
        # Strategy 4: Detect underline patterns in text
        underline_candidates = self._detect_underline_patterns(textract_blocks, page_number)
        candidates.extend(underline_candidates)
        logger.info(f"Detected {len(underline_candidates)} underline pattern candidates")
        
        # Strategy 5: Detect KEY_VALUE_SET blocks (Textract form analysis)
        # These are form fields that Textract has already identified
        key_value_candidates = self._detect_key_value_fields(textract_blocks, block_map, text_blocks, page_number)
        candidates.extend(key_value_candidates)
        logger.info(f"Detected {len(key_value_candidates)} KEY_VALUE_SET candidates")
        
        # Deduplicate candidates with overlapping bounding boxes
        candidates = self._deduplicate_candidates(candidates)
        
        # Attach nearby context to all candidates
        for candidate in candidates:
            self._attach_context(candidate, text_blocks)
        
        logger.info(f"Total field candidates detected: {len(candidates)}")
        return candidates
    
    def _generate_candidate_id(self) -> str:
        """Generate unique candidate ID."""
        self._candidate_counter += 1
        return f"field_{self._candidate_counter:04d}"
    
    def _extract_text_blocks(self, textract_blocks: List[Dict[str, Any]]) -> List[TextBlock]:
        """Extract text blocks with spatial info for context matching."""
        text_blocks = []
        
        for block in textract_blocks:
            block_type = block.get('BlockType', '')
            
            if block_type in ('WORD', 'LINE') and block.get('Text'):
                geometry = block.get('Geometry', {})
                bbox_dict = geometry.get('BoundingBox', {})
                
                if bbox_dict:
                    text_blocks.append(TextBlock(
                        text=block.get('Text', ''),
                        bounding_box=BoundingBox.from_textract(bbox_dict),
                        block_type=block_type,
                        confidence=block.get('Confidence', 0),
                        block_id=block.get('Id', '')
                    ))
        
        return text_blocks
    
    def _detect_checkboxes(
        self,
        textract_blocks: List[Dict[str, Any]],
        block_map: Dict[str, Dict[str, Any]],
        page_number: int
    ) -> List[FieldCandidate]:
        """
        Detect checkboxes from SELECTION_ELEMENT blocks.
        
        Checkboxes are considered empty if:
        - SelectionStatus is NOT_SELECTED
        - Or if selection cannot be determined (ambiguous)
        """
        candidates = []
        
        for block in textract_blocks:
            if block.get('BlockType') != 'SELECTION_ELEMENT':
                continue
            
            geometry = block.get('Geometry', {})
            bbox_dict = geometry.get('BoundingBox', {})
            
            if not bbox_dict:
                continue
            
            selection_status = block.get('SelectionStatus', 'NOT_SELECTED')
            
            # Only detect unchecked or ambiguous checkboxes as empty
            # Checked checkboxes are not "empty" fields
            if selection_status == 'SELECTED':
                continue
            
            bbox = BoundingBox.from_textract(bbox_dict)
            
            candidates.append(FieldCandidate(
                candidate_id=self._generate_candidate_id(),
                field_type=FieldType.CHECKBOX,
                bounding_box=bbox,
                page_number=page_number,
                detection_method="selection_element",
                detection_confidence=block.get('Confidence', 0) / 100.0,
                selection_status=selection_status
            ))
        
        return candidates
    
    def _detect_empty_table_cells(
        self,
        textract_blocks: List[Dict[str, Any]],
        block_map: Dict[str, Dict[str, Any]],
        page_number: int
    ) -> List[FieldCandidate]:
        """
        Detect empty table cells.
        
        A cell is empty if it has no child WORD blocks with text content.
        """
        candidates = []
        
        for block in textract_blocks:
            if block.get('BlockType') != 'CELL':
                continue
            
            geometry = block.get('Geometry', {})
            bbox_dict = geometry.get('BoundingBox', {})
            
            if not bbox_dict:
                continue
            
            # Check if cell has any text content
            has_text = False
            relationships = block.get('Relationships', [])
            
            for rel in relationships:
                if rel.get('Type') == 'CHILD':
                    for child_id in rel.get('Ids', []):
                        child_block = block_map.get(child_id)
                        if child_block and child_block.get('BlockType') == 'WORD':
                            if child_block.get('Text', '').strip():
                                has_text = True
                                break
                    if has_text:
                        break
            
            if has_text:
                continue
            
            bbox = BoundingBox.from_textract(bbox_dict)
            
            # Get table context
            table_id = None
            row_index = block.get('RowIndex')
            col_index = block.get('ColumnIndex')
            
            # Find parent table
            for rel in relationships:
                if rel.get('Type') == 'TABLE':
                    table_ids = rel.get('Ids', [])
                    if table_ids:
                        table_id = table_ids[0]
            
            candidates.append(FieldCandidate(
                candidate_id=self._generate_candidate_id(),
                field_type=FieldType.TABLE_CELL,
                bounding_box=bbox,
                page_number=page_number,
                detection_method="empty_cell",
                detection_confidence=block.get('Confidence', 0) / 100.0,
                table_id=table_id,
                row_index=row_index,
                col_index=col_index
            ))
        
        return candidates
    
    def _detect_geometry_lines(
        self,
        textract_blocks: List[Dict[str, Any]],
        text_blocks: List[TextBlock],
        page_number: int
    ) -> List[FieldCandidate]:
        """
        Detect horizontal fill lines using geometry analysis.
        
        Strategy:
        - Find LINE blocks with high aspect ratio (wide and thin)
        - Ensure no significant text overlap (it's an empty line, not text)
        - Validate against minimum size thresholds
        
        Tradeoff: This method detects visual lines that may be underlines
        for form fields. It may produce false positives for decorative lines.
        """
        candidates = []
        
        for block in textract_blocks:
            if block.get('BlockType') != 'LINE':
                continue
            
            geometry = block.get('Geometry', {})
            bbox_dict = geometry.get('BoundingBox', {})
            
            if not bbox_dict:
                continue
            
            bbox = BoundingBox.from_textract(bbox_dict)
            
            # Check geometric criteria for a fill line
            if not self._is_fill_line_geometry(bbox):
                continue
            
            # Check if this line overlaps significantly with text
            # (if so, it's likely text content, not a fill line)
            text_content = block.get('Text', '').strip()
            if text_content and not self._is_underline_text(text_content):
                continue
            
            candidates.append(FieldCandidate(
                candidate_id=self._generate_candidate_id(),
                field_type=FieldType.LINE,
                bounding_box=bbox,
                page_number=page_number,
                detection_method="geometry_line",
                detection_confidence=0.7  # Lower confidence for geometric detection
            ))
        
        return candidates
    
    def _is_fill_line_geometry(self, bbox: BoundingBox) -> bool:
        """
        Check if bounding box geometry suggests a fill line.
        
        Criteria:
        1. High aspect ratio (wide and thin)
        2. Minimum width (not too small)
        3. Maximum height (not too tall)
        """
        if bbox.aspect_ratio < self.MIN_LINE_ASPECT_RATIO:
            return False
        
        if bbox.width < self.MIN_LINE_WIDTH:
            return False
        
        if bbox.height > self.MAX_LINE_HEIGHT:
            return False
        
        return True
    
    def _is_underline_text(self, text: str) -> bool:
        """Check if text consists primarily of underline characters."""
        underline_chars = set('_-—–')
        text_stripped = text.replace(' ', '')
        
        if not text_stripped:
            return False
        
        underline_count = sum(1 for c in text_stripped if c in underline_chars)
        return underline_count / len(text_stripped) > 0.7
    
    def _detect_underline_patterns(
        self,
        textract_blocks: List[Dict[str, Any]],
        page_number: int
    ) -> List[FieldCandidate]:
        """
        Detect fill fields from underline text patterns (e.g., "____").
        
        This catches fields that Textract reads as text containing underscores,
        which often indicates a fill-in-the-blank field.
        """
        candidates = []
        
        for block in textract_blocks:
            if block.get('BlockType') not in ('WORD', 'LINE'):
                continue
            
            text = block.get('Text', '').strip()
            
            if not text:
                continue
            
            # Check for underline pattern
            if not self._is_underline_text(text):
                continue
            
            # Must meet minimum length
            if len(text.replace(' ', '')) < self.UNDERLINE_PATTERN_MIN_LENGTH:
                continue
            
            geometry = block.get('Geometry', {})
            bbox_dict = geometry.get('BoundingBox', {})
            
            if not bbox_dict:
                continue
            
            bbox = BoundingBox.from_textract(bbox_dict)
            
            candidates.append(FieldCandidate(
                candidate_id=self._generate_candidate_id(),
                field_type=FieldType.LINE,
                bounding_box=bbox,
                page_number=page_number,
                detection_method="underline_pattern",
                detection_confidence=0.85  # Higher confidence for explicit underlines
            ))
        
        return candidates
    
    def _detect_key_value_fields(
        self,
        textract_blocks: List[Dict[str, Any]],
        block_map: Dict[str, Dict[str, Any]],
        text_blocks: List[TextBlock],
        page_number: int
    ) -> List[FieldCandidate]:
        """
        Detect form fields from KEY_VALUE_SET blocks.
        
        KEY_VALUE_SET blocks are created by Textract's analyze_document with
        FeatureTypes=['FORMS']. They represent detected form fields with:
        - KEY: The label/question
        - VALUE: The input area (what we want to detect)
        
        This is a highly reliable detection method since Textract has already
        identified these as form fields using its ML models.
        """
        candidates = []
        
        for block in textract_blocks:
            if block.get('BlockType') != 'KEY_VALUE_SET':
                continue
            
            entity_types = block.get('EntityTypes', [])
            
            # We want VALUE blocks - these are the input areas
            if 'VALUE' not in entity_types:
                continue
            
            geometry = block.get('Geometry', {})
            bbox_dict = geometry.get('BoundingBox', {})
            
            if not bbox_dict:
                continue
            
            bbox = BoundingBox.from_textract(bbox_dict)
            
            # Extract value text if present (for already-filled fields)
            value_text = ''
            relationships = block.get('Relationships', [])
            for rel in relationships:
                if rel.get('Type') == 'CHILD':
                    for child_id in rel.get('Ids', []):
                        child_block = block_map.get(child_id)
                        if child_block and child_block.get('BlockType') == 'WORD':
                            word = child_block.get('Text', '').strip()
                            if word:
                                value_text += ' ' + word
            value_text = value_text.strip()
            
            # Detect the key (label) text by finding nearest text to the left or above
            supporting_text = self._find_key_for_value(bbox, text_blocks)
            
            # Determine if this is an empty field vs filled
            # Empty fields have no value text or only underscores/placeholders
            is_empty = not value_text or self._is_underline_text(value_text)
            
            # We still create candidates for filled fields (for display purposes)
            # but mark them with lower confidence
            detection_confidence = 0.9 if is_empty else 0.7
            
            candidates.append(FieldCandidate(
                candidate_id=self._generate_candidate_id(),
                field_type=FieldType.LINE,
                bounding_box=bbox,
                page_number=page_number,
                detection_method="key_value_set",
                detection_confidence=detection_confidence,
                supporting_text=supporting_text
            ))
        
        return candidates
    
    def _find_key_for_value(
        self,
        value_bbox: BoundingBox,
        text_blocks: List[TextBlock]
    ) -> str:
        """
        Find the most likely label (key) for a VALUE bounding box.
        
        Labels are typically:
        1. Directly to the left on the same row
        2. Above and left-aligned
        3. Above the field (stacked layout)
        
        Args:
            value_bbox: The VALUE block's bounding box
            text_blocks: All text blocks on the page
        
        Returns:
            The label text, or empty string if not found
        """
        best_label = None
        best_score = float('inf')
        
        for tb in text_blocks:
            # Calculate position relative to value box
            is_left = tb.bounding_box.right <= value_bbox.left + 0.02
            is_above = tb.bounding_box.bottom <= value_bbox.top + 0.02
            same_row = abs(tb.bounding_box.center_y - value_bbox.center_y) < 0.04  # Slightly relaxed
            horizontal_alignment = abs(tb.bounding_box.left - value_bbox.left)
            
            # Skip text to the right or below
            if tb.bounding_box.left > value_bbox.right + 0.05:
                continue
            if tb.bounding_box.top > value_bbox.bottom + 0.02:
                continue
            
            # Skip overlapping text
            if value_bbox.overlaps(tb.bounding_box, threshold=0.3):
                continue
            
            # Score based on position (lower = better)
            if is_left and same_row:
                # Best: directly to the left on same row
                horizontal_dist = value_bbox.left - tb.bounding_box.right
                score = horizontal_dist * 0.3
            elif is_above and horizontal_alignment < 0.05:
                # Good: above and well-aligned
                vertical_dist = value_bbox.top - tb.bounding_box.bottom
                score = vertical_dist * 0.8 + horizontal_alignment
            elif is_above and horizontal_alignment < 0.15:
                # OK: above and roughly aligned
                vertical_dist = value_bbox.top - tb.bounding_box.bottom
                score = vertical_dist + horizontal_alignment * 2
            elif is_left:
                # OK: to the left, different row
                horizontal_dist = value_bbox.left - tb.bounding_box.right
                vertical_dist = abs(tb.bounding_box.center_y - value_bbox.center_y)
                score = horizontal_dist * 0.8 + vertical_dist * 2
            elif is_above:
                # Fallback: above but not aligned
                vertical_dist = value_bbox.top - tb.bounding_box.bottom
                score = vertical_dist * 2 + horizontal_alignment * 3
            else:
                # Not a good label candidate
                continue
            
            # Update best match (relaxed threshold from 0.25 to 0.35)
            if score < best_score and score < 0.35:
                best_score = score
                best_label = tb.text
        
        return best_label or ""
    
    def _deduplicate_candidates(
        self,
        candidates: List[FieldCandidate]
    ) -> List[FieldCandidate]:
        """
        Remove duplicate candidates with significantly overlapping bounding boxes.
        
        When duplicates are found, prefer:
        1. Higher detection confidence
        2. More specific detection method (underline_pattern > geometry_line)
        """
        if len(candidates) <= 1:
            return candidates
        
        # Sort by confidence (descending) to prefer higher confidence
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (c.detection_confidence, c.detection_method == "underline_pattern"),
            reverse=True
        )
        
        deduplicated = []
        used_indices: Set[int] = set()
        
        for i, candidate in enumerate(sorted_candidates):
            if i in used_indices:
                continue
            
            deduplicated.append(candidate)
            
            # Mark all overlapping candidates as used
            for j, other in enumerate(sorted_candidates[i+1:], start=i+1):
                if j in used_indices:
                    continue
                
                if candidate.bounding_box.overlaps(other.bounding_box, threshold=0.5):
                    used_indices.add(j)
        
        return deduplicated
    
    def _attach_context(
        self,
        candidate: FieldCandidate,
        text_blocks: List[TextBlock]
    ) -> None:
        """
        Attach nearby text blocks as context for semantic classification.
        
        Context collection strategy:
        1. Find all text within spatial radius
        2. Prioritize text to the left (most likely to be labels)
        3. Include text above as secondary context
        4. Set supporting_text to the most likely label
        
        Improved scoring for legal forms where labels may be:
        - Directly to the left on same row (most common)
        - Above and left-aligned (column headers in tables)
        - Above the field (stacked form layout)
        - On a previous line but clearly associated
        """
        nearby = []
        best_label: Optional[TextBlock] = None
        best_label_score = float('inf')
        
        for tb in text_blocks:
            # Skip if too far away (use expanded radius for context)
            dx = abs(tb.bounding_box.center_x - candidate.bounding_box.center_x)
            dy = abs(tb.bounding_box.center_y - candidate.bounding_box.center_y)
            
            if dx > self.CONTEXT_RADIUS_X or dy > self.CONTEXT_RADIUS_Y:
                continue
            
            # Skip if significantly overlapping (might be part of the field)
            if candidate.bounding_box.overlaps(tb.bounding_box, threshold=0.3):
                continue
            
            nearby.append(tb)
            
            # Score as potential label
            # Labels are typically left-of or above the field
            is_left = tb.bounding_box.is_left_of(candidate.bounding_box)
            is_above = tb.bounding_box.is_above(candidate.bounding_box)
            same_row = tb.bounding_box.same_row(candidate.bounding_box)
            
            # Check if text is to the left of field start (even if overlapping vertically)
            is_left_of_start = tb.bounding_box.right <= candidate.bounding_box.left + 0.02
            
            # Check alignment for "above" labels
            horizontal_alignment = abs(tb.bounding_box.left - candidate.bounding_box.left)
            
            if is_left_of_start and same_row:
                # Best case: label directly to the left on same row
                horizontal_dist = candidate.bounding_box.left - tb.bounding_box.right
                score = horizontal_dist * 0.3  # Lower multiplier = higher priority
            elif is_left and same_row:
                # Good case: left and same row (maybe slightly overlapping)
                horizontal_dist = max(0, candidate.bounding_box.left - tb.bounding_box.right)
                score = horizontal_dist * 0.5 + 0.02
            elif is_above and horizontal_alignment < 0.05:
                # Good case: above and well-aligned (same column)
                vertical_dist = candidate.bounding_box.top - tb.bounding_box.bottom
                score = vertical_dist * 0.8 + horizontal_alignment
            elif is_above and horizontal_alignment < 0.15:
                # OK case: above and roughly aligned
                vertical_dist = candidate.bounding_box.top - tb.bounding_box.bottom
                score = vertical_dist + horizontal_alignment * 2
            elif is_left_of_start:
                # Label to the left, different row
                horizontal_dist = candidate.bounding_box.left - tb.bounding_box.right
                vertical_dist = abs(tb.bounding_box.center_y - candidate.bounding_box.center_y)
                score = horizontal_dist * 0.8 + vertical_dist * 2
            elif is_above:
                # Label above, not well aligned
                vertical_dist = candidate.bounding_box.top - tb.bounding_box.bottom
                score = vertical_dist * 2 + horizontal_alignment * 3
            else:
                # Not a good label candidate
                score = float('inf')
            
            # Accept labels with scores up to 0.3 (relaxed from 0.2)
            if score < best_label_score and score < 0.3:
                best_label_score = score
                best_label = tb
        
        # Sort nearby text by position (top-to-bottom, left-to-right)
        nearby.sort(key=lambda tb: (tb.bounding_box.top, tb.bounding_box.left))
        
        candidate.nearby_text = nearby
        candidate.supporting_text = best_label.text if best_label else ""

