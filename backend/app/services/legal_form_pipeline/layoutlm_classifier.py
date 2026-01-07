"""
LayoutLMv3 Semantic Classification Service
==========================================

Uses LayoutLMv3 for layout-aware semantic classification of form field candidates.

LayoutLMv3 Architecture:
------------------------
- Pre-trained on large-scale document understanding tasks
- Multimodal: combines text, layout (bounding boxes), and visual features
- Uses unified text-image pretraining (no separate CNN backbone)

Input Format:
-------------
- Tokens: tokenized text from field context
- Bounding boxes: [x0, y0, x1, y1] normalized to 0-1000 range
- Page image: optional but improves accuracy

Output:
-------
- Label prediction from closed ontology
- Confidence score
- Hidden states for ambiguity detection

Tradeoffs:
----------
1. Model size vs speed:
   - LayoutLMv3-base (125M params) - faster, good for most forms
   - LayoutLMv3-large (368M params) - more accurate, slower
   - We default to base for production throughput

2. Fine-tuning requirements:
   - Zero-shot works but has lower accuracy
   - Fine-tuned on legal form dataset recommended
   - We provide a pre-trained head option

3. Batch processing:
   - Batching improves throughput but increases memory
   - Single-field mode for maximum accuracy on ambiguous cases

IMPORTANT: This classifier is the PRIMARY source of semantic labels.
GLM-4.5V is only used for ambiguity resolution.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .ontology import SemanticLabel, LegalFieldOntology, ONTOLOGY
from .field_candidates import FieldCandidate, BoundingBox, TextBlock

logger = logging.getLogger(__name__)


# Check if transformers is available
TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from transformers import (
        LayoutLMv3Processor,
        LayoutLMv3ForSequenceClassification,
        LayoutLMv3Tokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning(
        "transformers library not available. "
        "LayoutLMClassifier will use fallback heuristic mode. "
        "Install with: pip install transformers torch"
    )


@dataclass
class ClassificationResult:
    """
    Result of semantic classification for a field candidate.
    
    Includes primary prediction and alternatives for ambiguity detection.
    """
    candidate_id: str
    primary_label: SemanticLabel
    primary_confidence: float
    
    # Alternative predictions for ambiguity detection
    # Sorted by confidence (descending)
    alternatives: List[Tuple[SemanticLabel, float]]
    
    # Is this prediction ambiguous? (requires GLM adjudication)
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    
    # Classification metadata
    method: str = "layoutlm"  # "layoutlm" or "heuristic_fallback"
    input_text: str = ""
    
    def __post_init__(self):
        """Compute ambiguity after initialization."""
        if not self.is_ambiguous:
            self.is_ambiguous, self.ambiguity_reason = self._check_ambiguity()
    
    def _check_ambiguity(self) -> Tuple[bool, Optional[str]]:
        """
        Determine if this classification is ambiguous.
        
        Ambiguity conditions:
        1. Multiple labels within 0.1 confidence of top
        2. Primary confidence below threshold
        3. Attestational field (legal significance)
        """
        # Condition 1: Close alternatives
        if self.alternatives:
            top_alternative_conf = self.alternatives[0][1]
            if self.primary_confidence - top_alternative_conf < 0.1:
                return True, f"Close alternative: {self.alternatives[0][0].value} ({top_alternative_conf:.2f})"
        
        # Condition 2: Low confidence
        if self.primary_confidence < 0.6:
            return True, f"Low confidence: {self.primary_confidence:.2f}"
        
        # Condition 3: Attestational field needs extra scrutiny
        if self.primary_label in ONTOLOGY.attestational_labels:
            if self.primary_confidence < 0.85:
                return True, f"Attestational field with confidence {self.primary_confidence:.2f}"
        
        return False, None


class LayoutLMClassifier:
    """
    LayoutLMv3-based semantic classifier for legal form fields.
    
    Modes:
    1. Model mode: Uses actual LayoutLMv3 model (requires GPU recommended)
    2. Heuristic fallback: Pattern matching when model unavailable
    
    The classifier normalizes all inputs to match LayoutLM's expected format:
    - Bounding boxes scaled to 0-1000 range
    - Text tokenized with proper special tokens
    - Image resized to expected dimensions
    
    IMPORTANT: The base LayoutLMv3 model does NOT have a fine-tuned classification
    head for legal form fields. Using it directly produces near-random outputs.
    Until a fine-tuned model is available, we default to heuristic mode which
    provides much better accuracy through pattern matching.
    """
    
    # Ambiguity threshold: if top-2 labels are within this confidence
    AMBIGUITY_CONFIDENCE_THRESHOLD = 0.1
    
    # Minimum confidence for a valid prediction
    MIN_CONFIDENCE_THRESHOLD = 0.4
    
    # LayoutLM bbox normalization
    LAYOUTLM_BBOX_SCALE = 1000
    
    # IMPORTANT: Set to True to require a fine-tuned model before using LayoutLM.
    # When True, base models (with random classification heads) will fall back to heuristics.
    # Set to False only if you have a fine-tuned model checkpoint.
    REQUIRE_FINETUNED_MODEL = True
    
    # Known base model names that should NOT be used for classification without fine-tuning
    BASE_MODEL_NAMES = [
        "microsoft/layoutlmv3-base",
        "microsoft/layoutlmv3-large",
        "microsoft/layoutlm-base-uncased",
        "microsoft/layoutlm-large-uncased",
        "microsoft/layoutlmv2-base-uncased",
        "microsoft/layoutlmv2-large-uncased",
    ]
    
    def __init__(
        self,
        model_name: str = "microsoft/layoutlmv3-base",
        use_gpu: bool = True,
        fallback_to_heuristics: bool = True,
        force_heuristic_mode: bool = False
    ):
        """
        Initialize the LayoutLM classifier.
        
        Args:
            model_name: HuggingFace model name or path to fine-tuned model
            use_gpu: Whether to use GPU if available
            fallback_to_heuristics: Use heuristics if model fails to load
            force_heuristic_mode: Force heuristic mode even if model is available
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.fallback_to_heuristics = fallback_to_heuristics
        self.force_heuristic_mode = force_heuristic_mode
        
        self.model = None
        self.processor = None
        self.device = None
        self.ontology = ONTOLOGY
        
        self._mode = "not_initialized"
        self._is_finetuned = False
        
        # Check if we should force heuristic mode
        if force_heuristic_mode:
            logger.info("Force heuristic mode enabled - using pattern matching")
            self._mode = "heuristic"
        elif TRANSFORMERS_AVAILABLE:
            self._initialize_model()
        elif fallback_to_heuristics:
            logger.warning("Using heuristic fallback mode (transformers not installed)")
            self._mode = "heuristic"
        else:
            raise RuntimeError(
                "transformers library required. Install with: pip install transformers torch"
            )
    
    def _initialize_model(self):
        """
        Initialize LayoutLMv3 model and processor.
        
        Uses sequence classification head for our label ontology.
        
        IMPORTANT: Base models like microsoft/layoutlmv3-base do NOT have
        fine-tuned classification heads for our ontology. Loading them with
        num_labels creates a RANDOM classification head that produces garbage
        outputs. We detect this and fall back to heuristics.
        """
        try:
            # Check if this is a base model without fine-tuning
            is_base_model = self._is_base_model(self.model_name)
            
            if is_base_model and self.REQUIRE_FINETUNED_MODEL:
                logger.warning(
                    f"Model '{self.model_name}' is a base model without fine-tuned "
                    f"classification head for legal form fields. Using base model would "
                    f"produce near-random outputs. Falling back to heuristic mode. "
                    f"To use LayoutLM, provide a fine-tuned model checkpoint or set "
                    f"REQUIRE_FINETUNED_MODEL = False (not recommended)."
                )
                self._mode = "heuristic"
                self._is_finetuned = False
                return
            
            logger.info(f"Loading LayoutLMv3 model: {self.model_name}")
            
            # Determine device
            if self.use_gpu and torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA GPU for inference")
            elif self.use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using Apple MPS for inference")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for inference")
            
            # Load processor (handles tokenization and bbox normalization)
            self.processor = LayoutLMv3Processor.from_pretrained(
                self.model_name,
                apply_ocr=False  # We provide our own OCR from Textract
            )
            
            # Load model with classification head
            # num_labels matches our ontology size
            self.model = LayoutLMv3ForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.ontology.num_labels
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            # Mark as fine-tuned if we got here (not a base model)
            self._is_finetuned = not is_base_model
            self._mode = "model"
            logger.info(f"LayoutLMv3 loaded successfully. Labels: {self.ontology.num_labels}, Fine-tuned: {self._is_finetuned}")
            
        except Exception as e:
            logger.error(f"Failed to load LayoutLMv3 model: {e}")
            if self.fallback_to_heuristics:
                logger.warning("Falling back to heuristic classification")
                self._mode = "heuristic"
            else:
                raise
    
    def _is_base_model(self, model_name: str) -> bool:
        """
        Check if the model name refers to a base (non-fine-tuned) model.
        
        Base models have randomly initialized classification heads when loaded
        with a custom num_labels, making them unsuitable for classification
        without fine-tuning.
        
        Args:
            model_name: HuggingFace model name or local path
        
        Returns:
            True if this is a known base model that shouldn't be used directly
        """
        # Check against known base model names
        model_name_lower = model_name.lower()
        for base_name in self.BASE_MODEL_NAMES:
            if base_name.lower() in model_name_lower or model_name_lower == base_name.lower():
                return True
        
        # If it's a local path, check if it contains indicators of fine-tuning
        # Fine-tuned models typically have "finetuned", "legal", "form" in path
        finetuned_indicators = ["finetuned", "fine-tuned", "legal", "form", "trained"]
        if any(ind in model_name_lower for ind in finetuned_indicators):
            return False
        
        # If it's not a recognized HuggingFace base model, assume it might be fine-tuned
        if "/" not in model_name or not model_name.startswith("microsoft/"):
            return False
        
        return False
    
    @property
    def mode(self) -> str:
        """Current classification mode: 'model' or 'heuristic'."""
        return self._mode
    
    @property
    def is_finetuned(self) -> bool:
        """Whether the loaded model has a fine-tuned classification head."""
        return self._is_finetuned
    
    @property
    def mode_description(self) -> str:
        """Human-readable description of the current mode."""
        if self._mode == "model":
            if self._is_finetuned:
                return "LayoutLMv3 (fine-tuned)"
            else:
                return "LayoutLMv3 (base - WARNING: not fine-tuned)"
        elif self._mode == "heuristic":
            return "Heuristic pattern matching"
        else:
            return f"Unknown ({self._mode})"
    
    def classify_candidates(
        self,
        candidates: List[FieldCandidate],
        page_image: Optional[Any] = None
    ) -> List[ClassificationResult]:
        """
        Classify a batch of field candidates.
        
        Args:
            candidates: List of FieldCandidate objects with context
            page_image: Optional PIL Image of the page (improves accuracy)
        
        Returns:
            List of ClassificationResult objects
        """
        if self._mode == "model":
            return self._classify_with_model(candidates, page_image)
        else:
            return self._classify_with_heuristics(candidates)
    
    def classify_single(
        self,
        candidate: FieldCandidate,
        page_image: Optional[Any] = None
    ) -> ClassificationResult:
        """
        Classify a single field candidate.
        
        Useful for ambiguous cases requiring maximum accuracy.
        
        Args:
            candidate: FieldCandidate object with context
            page_image: Optional PIL Image of the page
        
        Returns:
            ClassificationResult object
        """
        results = self.classify_candidates([candidate], page_image)
        return results[0] if results else self._create_unknown_result(candidate.candidate_id)
    
    def _classify_with_model(
        self,
        candidates: List[FieldCandidate],
        page_image: Optional[Any] = None
    ) -> List[ClassificationResult]:
        """
        Classify using the LayoutLMv3 model.
        
        Batch processing for efficiency.
        """
        results = []
        
        for candidate in candidates:
            try:
                # Prepare input
                words, boxes = self._prepare_layoutlm_input(candidate)
                
                if not words:
                    # No context available - classify as unknown
                    results.append(self._create_unknown_result(
                        candidate.candidate_id,
                        reason="No text context available"
                    ))
                    continue
                
                # Prepare encoding
                encoding = self.processor(
                    text=words,
                    boxes=boxes,
                    images=page_image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move to device
                encoding = {k: v.to(self.device) for k, v in encoding.items()}
                
                # Run inference
                with torch.no_grad():
                    outputs = self.model(**encoding)
                
                # Get probabilities
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                
                # Get top predictions
                top_indices = np.argsort(probs)[::-1]
                
                primary_idx = top_indices[0]
                primary_label = self.ontology.get_label_from_id(primary_idx)
                primary_confidence = float(probs[primary_idx])
                
                # Get alternatives
                alternatives = []
                for idx in top_indices[1:6]:  # Top 5 alternatives
                    alt_label = self.ontology.get_label_from_id(idx)
                    alt_conf = float(probs[idx])
                    if alt_conf > 0.05:  # Only include meaningful alternatives
                        alternatives.append((alt_label, alt_conf))
                
                results.append(ClassificationResult(
                    candidate_id=candidate.candidate_id,
                    primary_label=primary_label,
                    primary_confidence=primary_confidence,
                    alternatives=alternatives,
                    method="layoutlm",
                    input_text=candidate.supporting_text
                ))
                
            except Exception as e:
                logger.error(f"Model classification failed for {candidate.candidate_id}: {e}")
                results.append(self._classify_with_heuristics([candidate])[0])
        
        return results
    
    def _classify_with_heuristics(
        self,
        candidates: List[FieldCandidate]
    ) -> List[ClassificationResult]:
        """
        Fallback heuristic classification using pattern matching.
        
        This provides reasonable accuracy without the model but should
        not be used for production legal work without review.
        """
        results = []
        
        for candidate in candidates:
            text = (candidate.supporting_text or "").lower().strip()
            context = candidate.get_context_text(max_chars=300).lower()
            
            # Try direct alias lookup first
            label = self.ontology.lookup_by_alias(text)
            if label:
                results.append(ClassificationResult(
                    candidate_id=candidate.candidate_id,
                    primary_label=label,
                    primary_confidence=0.85,
                    alternatives=[],
                    method="heuristic_fallback",
                    input_text=text
                ))
                continue
            
            # Pattern-based classification
            label, confidence, alternatives = self._heuristic_classify(text, context, candidate)
            
            results.append(ClassificationResult(
                candidate_id=candidate.candidate_id,
                primary_label=label,
                primary_confidence=confidence,
                alternatives=alternatives,
                method="heuristic_fallback",
                input_text=text
            ))
        
        return results
    
    def _heuristic_classify(
        self,
        text: str,
        context: str,
        candidate: FieldCandidate
    ) -> Tuple[SemanticLabel, float, List[Tuple[SemanticLabel, float]]]:
        """
        Pattern-based heuristic classification.
        
        Enhanced with comprehensive patterns for legal form fields.
        Returns (label, confidence, alternatives).
        """
        from .field_candidates import FieldType
        
        alternatives = []
        combined = f"{text} {context}".lower()
        
        # === CHECKBOX-SPECIFIC PATTERNS ===
        if candidate.field_type == FieldType.CHECKBOX:
            return self._classify_checkbox(text, context)
        
        # === NAME PATTERNS (expanded) ===
        if any(p in text for p in ['name', 'nombre', 'nm', 'n a m e']):
            if any(p in text for p in ['first', 'given', 'f name', 'fname', 'forename']):
                return SemanticLabel.FIRST_NAME, 0.9, []
            if any(p in text for p in ['last', 'surname', 'family', 'l name', 'lname']):
                return SemanticLabel.LAST_NAME, 0.9, []
            if any(p in text for p in ['middle', 'mi', 'm name', 'mname']):
                return SemanticLabel.MIDDLE_NAME, 0.9, []
            if any(p in combined for p in ['firm', 'law office', 'llp', 'llc', 'pllc', 'p.c.']):
                return SemanticLabel.LAW_FIRM_NAME, 0.85, [(SemanticLabel.PERSON_NAME, 0.1)]
            if any(p in combined for p in ['attorney', 'counsel', 'atty', 'esq', 'lawyer']):
                return SemanticLabel.ATTORNEY_NAME, 0.85, [(SemanticLabel.PERSON_NAME, 0.1)]
            if any(p in combined for p in ['plaintiff', 'claimant', 'complainant']):
                return SemanticLabel.PLAINTIFF_NAME, 0.8, [(SemanticLabel.PARTY_NAME, 0.15)]
            if any(p in combined for p in ['defendant', 'respondent to complaint']):
                return SemanticLabel.DEFENDANT_NAME, 0.8, [(SemanticLabel.PARTY_NAME, 0.15)]
            if any(p in text for p in ['petitioner', 'appellant']):
                return SemanticLabel.PETITIONER_NAME, 0.8, []
            if any(p in text for p in ['respondent', 'appellee']):
                return SemanticLabel.RESPONDENT_NAME, 0.8, []
            if any(p in combined for p in ['witness', 'deponent']):
                return SemanticLabel.WITNESS_NAME, 0.8, []
            if any(p in combined for p in ['judge', 'judicial officer', 'magistrate']):
                return SemanticLabel.JUDGE_NAME, 0.8, []
            # Generic name - check context for party indicators
            if 'v.' in context or 'vs' in context or 'versus' in context:
                return SemanticLabel.PARTY_NAME, 0.7, [(SemanticLabel.PERSON_NAME, 0.15)]
            return SemanticLabel.PERSON_NAME, 0.75, [(SemanticLabel.PARTY_NAME, 0.1)]
        
        # === ATTORNEY-SPECIFIC PATTERNS (without "name" keyword) ===
        if any(p in text for p in ['atty', 'attorney', 'counsel', 'esq', 'esquire']):
            if any(p in text for p in ['bar', 'sbn', 'license', '#', 'no.']):
                return SemanticLabel.ATTORNEY_BAR_NUMBER, 0.9, []
            if 'for' in text:  # "Attorney for Plaintiff"
                return SemanticLabel.ATTORNEY_NAME, 0.85, []
            return SemanticLabel.ATTORNEY_NAME, 0.75, [(SemanticLabel.LAW_FIRM_NAME, 0.15)]
        
        # === CASE/COURT PATTERNS (expanded) ===
        if any(p in text for p in ['case', 'matter', 'action']):
            if any(p in text for p in ['no', '#', 'number', 'num', ':', 'file']):
                return SemanticLabel.CASE_NUMBER, 0.9, []
        if any(p in text for p in ['docket', 'index']):
            return SemanticLabel.DOCKET_NUMBER, 0.9, []
        if any(p in text for p in ['court', 'tribunal']):
            if 'address' in text:
                return SemanticLabel.COURT_ADDRESS, 0.85, []
            if any(p in text for p in ['department', 'dept', 'div']):
                return SemanticLabel.COURT_DEPARTMENT, 0.85, []
            if any(p in text for p in ['division']):
                return SemanticLabel.COURT_DIVISION, 0.85, []
            return SemanticLabel.COURT_NAME, 0.8, [(SemanticLabel.JUDICIAL_DISTRICT, 0.1)]
        if any(p in text for p in ['county', 'parish']):
            return SemanticLabel.COUNTY, 0.85, []
        if any(p in text for p in ['district', 'circuit', 'jurisdiction']):
            return SemanticLabel.JUDICIAL_DISTRICT, 0.8, []
        
        # === DATE PATTERNS (expanded) ===
        if any(p in text for p in ['date', 'dated', 'dt', 'd a t e']):
            if any(p in combined for p in ['birth', 'dob', 'born', 'b-day', 'bday']):
                return SemanticLabel.DATE_OF_BIRTH, 0.9, []
            if any(p in text for p in ['filing', 'filed', 'file date']):
                return SemanticLabel.FILING_DATE, 0.9, []
            if any(p in text for p in ['hearing', 'trial', 'appear']):
                return SemanticLabel.HEARING_DATE, 0.85, []
            if any(p in text for p in ['incident', 'accident', 'injury', 'occurrence']):
                return SemanticLabel.INCIDENT_DATE, 0.85, []
            if any(p in text for p in ['service', 'served']):
                return SemanticLabel.SERVICE_DATE, 0.85, []
            if any(p in text for p in ['sign', 'signed', 'executed']):
                return SemanticLabel.DATE, 0.85, []
            return SemanticLabel.DATE, 0.8, []
        # Handle standalone "DOB" pattern
        if text.strip().upper() in ['DOB', 'D.O.B.', 'D.O.B', 'BIRTHDATE']:
            return SemanticLabel.DATE_OF_BIRTH, 0.9, []
        
        # === TIME PATTERNS ===
        if any(p in text for p in ['time', 'hour']):
            if 'hearing' in combined:
                return SemanticLabel.HEARING_TIME, 0.85, []
        
        # === SIGNATURE PATTERNS (expanded) ===
        if any(p in text for p in ['signature', 'sign', 'signed', 'autograph', 'x_']):
            if any(p in combined for p in ['notary', 'notarial', 'seal']):
                return SemanticLabel.NOTARY_SIGNATURE, 0.9, []
            if any(p in combined for p in ['commission', 'expires', 'expiration']):
                return SemanticLabel.NOTARY_COMMISSION_EXPIRY, 0.85, []
            return SemanticLabel.SIGNATURE, 0.9, []
        # Initial patterns
        if any(p in text for p in ['initial', 'initials']):
            return SemanticLabel.INITIALS, 0.85, []
        
        # === ADDRESS PATTERNS (expanded) ===
        if any(p in text for p in ['address', 'addr', 'adress']):
            if any(p in text for p in ['email', 'e-mail', 'electronic']):
                return SemanticLabel.EMAIL, 0.9, []
            if any(p in text for p in ['street', 'st.', 'ave', 'blvd', 'road', 'rd']):
                return SemanticLabel.STREET_ADDRESS, 0.85, []
            if any(p in text for p in ['mailing', 'home', 'residence', 'business']):
                return SemanticLabel.ADDRESS, 0.85, []
            return SemanticLabel.ADDRESS, 0.8, [(SemanticLabel.STREET_ADDRESS, 0.1)]
        if any(p in text for p in ['city', 'town', 'municipality']):
            return SemanticLabel.CITY, 0.85, []
        if 'state' in text and not any(p in text for p in ['bar', 'statement']):
            return SemanticLabel.STATE, 0.8, []
        if any(p in text for p in ['zip', 'postal', 'zip code', 'zipcode']):
            return SemanticLabel.ZIP_CODE, 0.85, []
        
        # === CONTACT PATTERNS (expanded) ===
        if any(p in text for p in ['phone', 'tel', 'telephone', 'cell', 'mobile', 'ph#', 'ph.', 'tel#', 'tel.']):
            return SemanticLabel.PHONE, 0.9, []
        if any(p in text for p in ['fax', 'facsimile', 'fax#', 'fax.']):
            return SemanticLabel.FAX, 0.9, []
        if any(p in text for p in ['email', 'e-mail', 'e mail', 'electronic mail']):
            return SemanticLabel.EMAIL, 0.9, []
        
        # === BAR NUMBER PATTERNS ===
        if any(p in text for p in ['bar', 'sbn', 'state bar']):
            if any(p in text for p in ['no', '#', 'number', 'num', ':']):
                return SemanticLabel.ATTORNEY_BAR_NUMBER, 0.9, []
        
        # === AMOUNT/MONETARY PATTERNS (expanded) ===
        if any(p in text for p in ['amount', 'amt', '$', 'sum', 'total']):
            if any(p in combined for p in ['damage', 'damages', 'award']):
                return SemanticLabel.DAMAGES_AMOUNT, 0.85, []
            if any(p in combined for p in ['bond', 'bail']):
                return SemanticLabel.BOND_AMOUNT, 0.85, []
            return SemanticLabel.AMOUNT, 0.8, []
        if any(p in text for p in ['fee', 'cost', 'payment']):
            if any(p in combined for p in ['filing', 'court', 'clerk']):
                return SemanticLabel.FILING_FEE, 0.85, []
            return SemanticLabel.FILING_FEE, 0.75, [(SemanticLabel.AMOUNT, 0.15)]
        
        # === NARRATIVE/DESCRIPTIVE PATTERNS (expanded) ===
        if any(p in text for p in ['description', 'describe', 'explain']):
            return SemanticLabel.DESCRIPTION, 0.75, []
        if 'cause' in text and any(p in combined for p in ['action', 'claim']):
            return SemanticLabel.CAUSE_OF_ACTION, 0.85, []
        if any(p in text for p in ['relief', 'remedy', 'prayer']):
            return SemanticLabel.RELIEF_REQUESTED, 0.8, []
        if any(p in text for p in ['facts', 'allegation', 'statement of']):
            return SemanticLabel.FACTS, 0.75, []
        if any(p in text for p in ['additional', 'other', 'comments', 'notes']):
            return SemanticLabel.ADDITIONAL_INFORMATION, 0.7, []
        
        # === DOCUMENT REFERENCE PATTERNS ===
        if any(p in text for p in ['exhibit', 'exh', 'ex.']):
            return SemanticLabel.EXHIBIT_NUMBER, 0.85, []
        if any(p in text for p in ['page', 'pg', 'p.']):
            if any(p in text for p in ['no', '#', 'number', 'of']):
                return SemanticLabel.PAGE_NUMBER, 0.8, []
        if any(p in text for p in ['attachment', 'attach', 'addendum']):
            return SemanticLabel.ATTACHMENT_NUMBER, 0.8, []
        
        # === DECLARATION/CERTIFICATION PATTERNS ===
        if any(p in combined for p in ['under penalty of perjury', 'penalty of perjury']):
            return SemanticLabel.UNDER_PENALTY_OF_PERJURY, 0.9, []
        if any(p in combined for p in ['declaration', 'declare', 'certify', 'affirm']):
            return SemanticLabel.DECLARATION_CONFIRMATION, 0.75, []
        
        # === PARTY TYPE PATTERNS (context-based) ===
        # Check if the field is near party designation without explicit "name"
        if 'plaintiff' in combined and not any(p in text for p in ['name', 'attorney']):
            return SemanticLabel.PLAINTIFF_NAME, 0.7, [(SemanticLabel.PARTY_NAME, 0.15)]
        if 'defendant' in combined and not any(p in text for p in ['name', 'attorney']):
            return SemanticLabel.DEFENDANT_NAME, 0.7, [(SemanticLabel.PARTY_NAME, 0.15)]
        
        # === CONTEXT-BASED PATTERNS ===
        # If we see strong context clues but no direct match
        if any(p in context for p in ['plaintiff', 'claimant']) and not text:
            return SemanticLabel.PLAINTIFF_NAME, 0.6, [(SemanticLabel.PARTY_NAME, 0.2)]
        if any(p in context for p in ['defendant']) and not text:
            return SemanticLabel.DEFENDANT_NAME, 0.6, [(SemanticLabel.PARTY_NAME, 0.2)]
        
        # === FALLBACK ===
        return SemanticLabel.UNKNOWN_FIELD, 0.4, []
    
    def _classify_checkbox(
        self,
        text: str,
        context: str
    ) -> Tuple[SemanticLabel, float, List[Tuple[SemanticLabel, float]]]:
        """
        Classify checkbox-type fields.
        
        Checkboxes often require more context since they represent
        attestations or selections. Enhanced with comprehensive patterns.
        """
        combined = f"{text} {context}".lower()
        
        # Age attestation - various phrasings
        if any(p in combined for p in ['18', 'eighteen', 'legal age']):
            if any(p in combined for p in ['over', 'older', 'above', 'at least', 'age of', 'years']):
                return SemanticLabel.AGE_OVER_18_CONFIRMATION, 0.85, []
        if any(p in combined for p in ['adult', 'majority', 'minor']):
            return SemanticLabel.AGE_OVER_18_CONFIRMATION, 0.75, []
        
        # Attorney representation - various phrasings
        if any(p in combined for p in ['attorney', 'counsel', 'lawyer', 'represented by']):
            if any(p in combined for p in ['represent', 'have', 'retained', 'by']):
                return SemanticLabel.CHECKBOX_REPRESENTED_BY_ATTORNEY, 0.8, []
        
        # Pro se / self-representation
        if any(p in combined for p in ['pro se', 'pro-se', 'propria persona', 'in propria']):
            return SemanticLabel.CHECKBOX_PRO_SE, 0.85, []
        if any(p in combined for p in ['without attorney', 'self-represent', 'representing myself']):
            return SemanticLabel.CHECKBOX_PRO_SE, 0.85, []
        if any(p in combined for p in ['self', 'myself']) and any(p in combined for p in ['represent', 'appearing']):
            return SemanticLabel.CHECKBOX_PRO_SE, 0.8, []
        
        # Fee waiver
        if any(p in combined for p in ['fee waiver', 'waive fee', 'waiver of fee', 'exempt from fee']):
            return SemanticLabel.CHECKBOX_FEE_WAIVER, 0.85, []
        if 'waive' in combined and 'fee' in combined:
            return SemanticLabel.CHECKBOX_FEE_WAIVER, 0.85, []
        if any(p in combined for p in ['in forma pauperis', 'ifp', 'indigent']):
            return SemanticLabel.CHECKBOX_FEE_WAIVER, 0.85, []
        
        # Interpreter needs
        if any(p in combined for p in ['interpreter', 'translation', 'translator', 'language assistance']):
            return SemanticLabel.CHECKBOX_INTERPRETER_NEEDED, 0.85, []
        if any(p in combined for p in ['spanish', 'chinese', 'vietnamese', 'korean', 'tagalog']) and 'speak' in combined:
            return SemanticLabel.CHECKBOX_INTERPRETER_NEEDED, 0.8, []
        
        # Disability accommodation
        if any(p in combined for p in ['disability', 'accommodation', 'accessible', 'ada', 'impairment']):
            return SemanticLabel.CHECKBOX_DISABILITY_ACCOMMODATION, 0.8, []
        if any(p in combined for p in ['wheelchair', 'hearing impaired', 'visually impaired', 'deaf']):
            return SemanticLabel.CHECKBOX_DISABILITY_ACCOMMODATION, 0.8, []
        
        # Consent patterns
        if any(p in combined for p in ['consent', 'agree', 'i agree', 'agreeing']):
            return SemanticLabel.CHECKBOX_CONSENT, 0.75, [(SemanticLabel.CHECKBOX_CERTIFICATION, 0.15)]
        if any(p in combined for p in ['authorize', 'permission', 'allow']):
            return SemanticLabel.CHECKBOX_CONSENT, 0.75, []
        
        # Certification/Declaration patterns
        if any(p in combined for p in ['certif', 'declar', 'attest', 'affirm', 'verify']):
            return SemanticLabel.CHECKBOX_CERTIFICATION, 0.75, [(SemanticLabel.CHECKBOX_CONSENT, 0.15)]
        if any(p in combined for p in ['true and correct', 'swear', 'oath']):
            return SemanticLabel.CHECKBOX_CERTIFICATION, 0.8, []
        
        # Perjury/Penalty patterns
        if any(p in combined for p in ['perjury', 'penalty', 'criminal penalties']):
            return SemanticLabel.UNDER_PENALTY_OF_PERJURY, 0.85, []
        if 'under penalty' in combined:
            return SemanticLabel.UNDER_PENALTY_OF_PERJURY, 0.85, []
        
        # Service-related checkboxes
        if any(p in combined for p in ['served', 'service', 'mail', 'mailed']):
            if any(p in combined for p in ['copy', 'document', 'papers']):
                return SemanticLabel.CHECKBOX_CERTIFICATION, 0.7, []
        
        # Document type selection (common in multi-form packets)
        if any(p in combined for p in ['complaint', 'answer', 'motion', 'petition', 'response']):
            return SemanticLabel.UNKNOWN_FIELD, 0.5, []  # Generic document selection
        
        # Court/jurisdiction selection
        if any(p in combined for p in ['superior court', 'district court', 'municipal', 'small claims']):
            return SemanticLabel.UNKNOWN_FIELD, 0.5, []
        
        # Unknown checkbox - but give slightly higher confidence if there's meaningful context
        if len(combined.strip()) > 10:
            return SemanticLabel.UNKNOWN_FIELD, 0.45, []
        
        return SemanticLabel.UNKNOWN_FIELD, 0.4, []
    
    def _prepare_layoutlm_input(
        self,
        candidate: FieldCandidate
    ) -> Tuple[List[str], List[List[int]]]:
        """
        Prepare input for LayoutLMv3.
        
        Converts candidate context to tokenized words with bounding boxes.
        Boxes are normalized to 0-1000 range as expected by LayoutLM.
        
        Returns:
            Tuple of (words, boxes) where boxes are [x0, y0, x1, y1] format
        """
        words = []
        boxes = []
        
        # Add supporting text (label) first
        if candidate.supporting_text:
            for word in candidate.supporting_text.split():
                words.append(word)
                # Use candidate's bounding box for label text
                # (approximation since we don't have exact word boxes)
                bbox = self._normalize_bbox(candidate.bounding_box)
                boxes.append(bbox)
        
        # Add nearby context
        for tb in candidate.nearby_text:
            if tb.text:
                for word in tb.text.split():
                    words.append(word)
                    bbox = self._normalize_bbox(tb.bounding_box)
                    boxes.append(bbox)
        
        return words, boxes
    
    def _normalize_bbox(self, bbox: BoundingBox) -> List[int]:
        """
        Normalize bounding box to LayoutLM format (0-1000 scale).
        
        Args:
            bbox: BoundingBox with 0-1 normalized coordinates
        
        Returns:
            [x0, y0, x1, y1] in 0-1000 scale
        """
        return [
            int(bbox.left * self.LAYOUTLM_BBOX_SCALE),
            int(bbox.top * self.LAYOUTLM_BBOX_SCALE),
            int(bbox.right * self.LAYOUTLM_BBOX_SCALE),
            int(bbox.bottom * self.LAYOUTLM_BBOX_SCALE)
        ]
    
    def _create_unknown_result(
        self,
        candidate_id: str,
        reason: str = ""
    ) -> ClassificationResult:
        """Create a result for unclassifiable fields."""
        return ClassificationResult(
            candidate_id=candidate_id,
            primary_label=SemanticLabel.UNKNOWN_FIELD,
            primary_confidence=0.4,
            alternatives=[],
            method="unknown",
            input_text="",
            is_ambiguous=True,
            ambiguity_reason=reason or "Classification not possible"
        )

