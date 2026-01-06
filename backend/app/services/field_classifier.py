"""
Field type classification service for form fields.
Classifies form field labels into predefined types or infers types dynamically.
"""
import logging
import re
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)


class FieldClassifier:
    """Service for classifying form field labels into field types."""
    
    # Predefined field types with their pattern matching rules
    # Patterns are ordered from most specific to least specific within each category
    # IMPORTANT: Patterns should be strict to avoid false positives
    FIELD_PATTERNS: Dict[str, List[str]] = {
        # Name fields - STRICT patterns only
        # Only match when "name" is clearly used as a field label for entering a person's name
        'name': [
            r'^name\s*$',  # Only matches standalone "Name" (no other words)
            r'^full\s+name\s*$',  # Only "Full Name" standalone
            r'^your\s+name\s*$',  # "Your Name" standalone
            r'^print\s+name\s*$',  # "Print Name" standalone
            r'^printed\s+name\s*$',  # "Printed Name" standalone
            r'^type\s+or\s+print\s+name\s*$',  # "Type or Print Name"
            r'^first\s+name\s*$',  # "First Name" standalone
            r'^last\s+name\s*$',  # "Last Name" standalone
            r'^middle\s+name\s*$',  # "Middle Name" standalone
            r'^first\s*,?\s*middle\s*,?\s*last\s*$',  # "First, Middle, Last"
        ],
        # Address fields - STRICT patterns
        'address': [
            r'^address\s*$',  # Standalone "Address"
            r'^street\s+address\s*$',  # "Street Address" standalone
            r'^mailing\s+address\s*$',  # "Mailing Address" standalone
            r'^residence\s+address\s*$',  # "Residence Address" standalone
            r'^home\s+address\s*$',  # "Home Address" standalone
            r'^address\s+line\s*\d*\s*$',  # "Address Line 1", "Address Line 2"
            r'^city\s*,?\s*state\s*,?\s*zip\s*$',  # "City, State, Zip"
            r'^zip\s+code\s*$',  # "Zip Code" standalone
            r'^postal\s+code\s*$',  # "Postal Code" standalone
            r'^city\s*$',  # "City" standalone
            r'^state\s*$',  # "State" standalone
        ],
        # Attorney fields - STRICT patterns
        'attorney': [
            r'^attorney\s*$',  # Standalone "Attorney"
            r'^attorney\s+name\s*$',  # "Attorney Name" standalone
            r'^attorney\s+for\s*$',  # "Attorney for" standalone
            r'^attorney\s+bar\s*#?\s*$',  # "Attorney Bar #"
        ],
        # Date fields - STRICT patterns
        'date': [
            r'^date\s*$',  # Standalone "Date"
            r'^date\s+of\s+birth\s*$',  # "Date of Birth" standalone
            r'^dob\s*$',  # "DOB" standalone
            r'^birth\s*date\s*$',  # "Birth Date" standalone
            r'^filing\s+date\s*$',  # "Filing Date" standalone
            r'^date\s+signed\s*$',  # "Date Signed" standalone
            r'^signature\s+date\s*$',  # "Signature Date" standalone
        ],
        # Phone fields - STRICT patterns
        'phone': [
            r'^phone\s*$',  # Standalone "Phone"
            r'^telephone\s*$',  # Standalone "Telephone"
            r'^phone\s+number\s*$',  # "Phone Number" standalone
            r'^telephone\s+number\s*$',  # "Telephone Number" standalone
            r'^cell\s+phone\s*$',  # "Cell Phone" standalone
            r'^mobile\s+phone\s*$',  # "Mobile Phone" standalone
            r'^contact\s+phone\s*$',  # "Contact Phone" standalone
            r'^fax\s*$',  # "Fax" standalone
            r'^fax\s+number\s*$',  # "Fax Number" standalone
        ],
        # Email fields - STRICT patterns
        'email': [
            r'^email\s*$',  # Standalone "Email"
            r'^e\s*-?\s*mail\s*$',  # "E-mail" or "Email" standalone
            r'^email\s+address\s*$',  # "Email Address" standalone
        ],
        # Signature fields - STRICT patterns
        'signature': [
            r'^signature\s*$',  # Standalone "Signature"
            r'^sign\s+here\s*$',  # "Sign Here" standalone
            r'^signed\s*$',  # "Signed" standalone
            r'^your\s+signature\s*$',  # "Your Signature"
        ],
        # Case number fields - STRICT patterns
        'case_number': [
            r'^case\s+number\s*$',  # "Case Number" standalone
            r'^case\s+no\.?\s*$',  # "Case No." standalone
            r'^case\s*#\s*$',  # "Case #" standalone
            r'^docket\s+number\s*$',  # "Docket Number" standalone
            r'^docket\s+no\.?\s*$',  # "Docket No." standalone
            r'^file\s+number\s*$',  # "File Number" standalone
        ],
        # Court fields - STRICT patterns
        'court': [
            r'^court\s*$',  # Standalone "Court"
            r'^court\s+name\s*$',  # "Court Name" standalone
            r'^court\s+location\s*$',  # "Court Location" standalone
            r'^judicial\s+district\s*$',  # "Judicial District" standalone
            r'^county\s*$',  # "County" standalone
        ],
        # Plaintiff fields - STRICT patterns
        'plaintiff': [
            r'^plaintiff\s*$',  # Standalone "Plaintiff"
        ],
        # Defendant fields - STRICT patterns
        'defendant': [
            r'^defendant\s*$',  # Standalone "Defendant"
        ],
        # Description fields - STRICT patterns
        'description': [
            r'^description\s*$',  # Standalone "Description"
        ],
        # Location fields - STRICT patterns
        'location': [
            r'^location\s*$',  # Standalone "Location"
        ],
        # Amount fields - STRICT patterns
        'amount': [
            r'^amount\s*$',  # Standalone "Amount"
            r'^total\s+amount\s*$',  # "Total Amount" standalone
        ],
    }
    
    def __init__(self):
        """Initialize field classifier."""
        # Compile regex patterns for efficiency
        self.compiled_patterns: Dict[str, List[re.Pattern]] = {}
        for field_type, patterns in self.FIELD_PATTERNS.items():
            self.compiled_patterns[field_type] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
        logger.info(f"Initialized FieldClassifier with {len(self.FIELD_PATTERNS)} field types")
    
    def classify_field(self, label_text: str) -> Tuple[str, float]:
        """
        Classify a form field label into a field type.
        
        Args:
            label_text: The label text from the form (e.g., "Name:", "Address")
        
        Returns:
            Tuple of (field_type, confidence) where:
            - field_type: The detected field type or 'other' if no strong match
            - confidence: Confidence score (0.0-1.0)
        """
        if not label_text or not label_text.strip():
            return ('unknown', 0.0)
        
        # Normalize the label text
        normalized = self._normalize_text(label_text)
        
        # Skip classification for very short or very long labels
        # These are unlikely to be meaningful field labels
        if len(normalized) < 2 or len(normalized) > 100:
            logger.debug(f"Skipping classification for label '{label_text}' - length {len(normalized)} outside reasonable range")
            return ('other', 0.3)
        
        # Track best match with highest confidence
        best_match = None
        best_confidence = 0.0
        
        # Check against predefined patterns
        # All patterns now use strict anchored matching (^ and $)
        for field_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                # Use fullmatch behavior - pattern must match the entire normalized text
                if pattern.fullmatch(normalized):
                    # High confidence for strict pattern matches
                    confidence = 0.95
                    if confidence > best_confidence:
                        best_match = field_type
                        best_confidence = confidence
                        logger.debug(f"Strict matched '{label_text}' to field type '{field_type}' with pattern '{pattern.pattern}'")
                        break  # Found a match for this field type, move on
        
        # Return best match if found
        if best_match:
            return (best_match, best_confidence)
        
        # No pattern match - return 'other' with low confidence
        # This prevents over-classification of fields
        logger.debug(f"No pattern match for '{label_text}' - classifying as 'other'")
        return ('other', 0.4)
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for pattern matching.
        
        Args:
            text: Raw text to normalize
        
        Returns:
            Normalized text (lowercase, punctuation removed/replaced)
        """
        # Convert to lowercase
        normalized = text.lower()
        
        # Replace common punctuation with spaces
        normalized = re.sub(r'[:;,\-\.\(\)]', ' ', normalized)
        
        # Replace multiple spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Strip leading/trailing whitespace
        normalized = normalized.strip()
        
        return normalized
    
    def _infer_field_type(self, normalized_text: str) -> str:
        """
        Infer a field type from normalized label text.
        
        Args:
            normalized_text: Normalized label text
        
        Returns:
            Inferred field type (normalized label text or simplified version)
        """
        # Remove common words that don't add meaning
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [w for w in normalized_text.split() if w not in stop_words]
        
        if not words:
            return normalized_text
        
        # Join words with underscore to create a field type identifier
        inferred = '_'.join(words)
        
        # Limit length to avoid overly long field types
        if len(inferred) > 50:
            inferred = inferred[:50]
        
        return inferred
    
    def classify_fields_batch(self, labels: List[str]) -> List[Tuple[str, str, float]]:
        """
        Classify multiple field labels in batch.
        
        Args:
            labels: List of label texts
        
        Returns:
            List of tuples: (label_text, field_type, confidence)
        """
        results = []
        for label in labels:
            field_type, confidence = self.classify_field(label)
            results.append((label, field_type, confidence))
        return results

