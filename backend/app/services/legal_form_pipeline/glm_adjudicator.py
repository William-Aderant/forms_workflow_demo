"""
GLM-4.5V Adjudicator Service
============================

Uses GLM-4.5V vision-language model ONLY for resolving ambiguous cases
from LayoutLM classification.

CRITICAL DESIGN PRINCIPLE:
--------------------------
GLM-4.5V is ADVISORY ONLY, not authoritative. It:
- CANNOT invent new labels (must choose from ontology)
- CANNOT override high-confidence LayoutLM predictions
- IS CALLED ONLY for ambiguous cases meeting specific criteria

Ambiguity Criteria (from pipeline spec):
----------------------------------------
1. Multiple labels score within 0.1 confidence of each other
2. Surrounding text exceeds 1 sentence (complex context)
3. Checkbox meaning is legal/attestational

Input to GLM:
-------------
- Cropped page image (region around field)
- Bounding box overlay highlighting the field
- Nearby text context
- Candidate labels from LayoutLM

Output from GLM:
----------------
- Best label FROM the provided ontology
- Short justification (stored for audit trail)
- Confidence adjustment (+/- modifier)

Legal Defensibility:
--------------------
All GLM adjudications are logged with:
- Input context
- Model reasoning
- Final decision rationale
This creates an audit trail for legal review.

Tradeoffs:
----------
1. Calling GLM adds latency (~2-5 seconds per field)
   - Mitigation: batch ambiguous fields, parallel calls
   - Only ~5-15% of fields typically need adjudication

2. GLM may "hallucinate" labels outside ontology
   - Mitigation: strict post-processing validation
   - Only accept labels matching ontology exactly

3. Determinism concerns:
   - Temperature=0 for reproducibility
   - Same input should produce same output
   - Log all inputs for reproducibility testing
"""

import logging
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from .ontology import SemanticLabel, LegalFieldOntology, ONTOLOGY
from .field_candidates import FieldCandidate, BoundingBox
from .layoutlm_classifier import ClassificationResult

logger = logging.getLogger(__name__)


class AdjudicationDecision(str, Enum):
    """Possible adjudication decisions."""
    ACCEPT_PRIMARY = "accept_primary"     # Keep LayoutLM's primary prediction
    ACCEPT_ALTERNATIVE = "accept_alternative"  # Switch to an alternative label
    MARK_UNKNOWN = "mark_unknown"         # Mark as unknown (couldn't resolve)
    INSUFFICIENT_CONTEXT = "insufficient_context"  # Not enough info to decide


@dataclass
class AdjudicationResult:
    """
    Result of GLM-4.5V adjudication for an ambiguous field.
    
    Contains full audit trail for legal review.
    """
    candidate_id: str
    original_label: SemanticLabel
    original_confidence: float
    
    # Adjudication outcome
    decision: AdjudicationDecision
    final_label: SemanticLabel
    final_confidence: float
    
    # GLM reasoning (for audit trail)
    glm_response: str = ""
    justification: str = ""
    
    # Confidence adjustment applied
    confidence_adjustment: float = 0.0
    
    # Audit metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    input_hash: str = ""  # Hash of input for reproducibility verification
    model_version: str = "glm-4.5v"
    
    # Processing metadata
    processing_time_ms: int = 0
    tokens_used: int = 0


class GLMAdjudicator:
    """
    GLM-4.5V adjudicator for resolving ambiguous field classifications.
    
    This service is designed to be called sparingly - only when LayoutLM
    cannot confidently classify a field.
    
    API Support:
    - OpenAI-compatible API (e.g., via vllm, ollama)
    - Direct ZhipuAI API
    - Local deployment
    """
    
    # System prompt enforcing ontology constraints
    SYSTEM_PROMPT = """You are a precise legal document analysis assistant. Your task is to classify form fields in legal documents.

CRITICAL RULES:
1. You MUST select a label from the provided ONTOLOGY ONLY
2. You CANNOT invent new labels
3. You MUST provide a brief justification
4. If uncertain, respond with "UNKNOWN_FIELD"

Output your response in this exact JSON format:
{
  "selected_label": "LABEL_FROM_ONTOLOGY",
  "justification": "Brief reason for selection (1-2 sentences)",
  "confidence_adjustment": 0.0
}

The confidence_adjustment should be:
- Positive (0.05 to 0.15) if you're more confident than the original classifier
- Negative (-0.15 to -0.05) if you're less confident
- Zero if confidence is similar"""

    # Prompt template for field adjudication
    ADJUDICATION_PROMPT_TEMPLATE = """Analyze this form field and select the most appropriate semantic label.

FIELD INFORMATION:
- Field Type: {field_type}
- Bounding Box: [{x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f}]
- Detection Confidence: {detection_confidence:.2f}

NEARBY TEXT CONTEXT:
{context_text}

SUPPORTING TEXT (likely label):
"{supporting_text}"

ORIGINAL CLASSIFICATION RESULT:
- Primary Label: {primary_label} (confidence: {primary_confidence:.2f})
- Alternative Labels: {alternatives}

AMBIGUITY REASON:
{ambiguity_reason}

VALID ONTOLOGY LABELS:
{ontology_labels}

Based on the visual context, text proximity, and legal document conventions, which label best describes this field?"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://open.bigmodel.cn/api/paas/v4",
        model_name: str = "glm-4v",
        timeout: int = 30,
        enable_logging: bool = True
    ):
        """
        Initialize the GLM adjudicator.
        
        Args:
            api_key: API key for GLM service (from env if not provided)
            api_base: Base URL for API
            model_name: Model name to use
            timeout: Request timeout in seconds
            enable_logging: Whether to log all adjudications
        """
        import os
        
        self.api_key = api_key or os.getenv('GLM_API_KEY') or os.getenv('ZHIPUAI_API_KEY')
        self.api_base = api_base
        self.model_name = model_name
        self.timeout = timeout
        self.enable_logging = enable_logging
        self.ontology = ONTOLOGY
        
        # Adjudication history for audit
        self._adjudication_log: List[AdjudicationResult] = []
        
        # Check for API availability
        self._api_available = bool(self.api_key)
        if not self._api_available:
            logger.warning(
                "GLM API key not configured. Adjudicator will use fallback mode. "
                "Set GLM_API_KEY or ZHIPUAI_API_KEY environment variable."
            )
    
    @property
    def is_available(self) -> bool:
        """Check if GLM API is available."""
        return self._api_available
    
    def should_adjudicate(
        self,
        candidate: FieldCandidate,
        classification: ClassificationResult
    ) -> bool:
        """
        Determine if a field classification needs GLM adjudication.
        
        Criteria:
        1. Classification is marked as ambiguous
        2. Field type is checkbox with legal significance
        3. Context text is complex (multiple sentences)
        
        Args:
            candidate: The field candidate
            classification: LayoutLM classification result
        
        Returns:
            True if adjudication is recommended
        """
        # Always adjudicate if marked ambiguous
        if classification.is_ambiguous:
            return True
        
        # Adjudicate attestational checkboxes below high confidence
        from .field_candidates import FieldType
        if candidate.field_type == FieldType.CHECKBOX:
            if classification.primary_label in self.ontology.attestational_labels:
                if classification.primary_confidence < 0.9:
                    return True
        
        # Adjudicate if context is complex (>1 sentence)
        context = candidate.get_context_text(max_chars=500)
        sentence_count = context.count('.') + context.count('?') + context.count('!')
        if sentence_count > 1 and classification.primary_confidence < 0.75:
            return True
        
        return False
    
    def adjudicate(
        self,
        candidate: FieldCandidate,
        classification: ClassificationResult,
        page_image: Optional[Any] = None
    ) -> AdjudicationResult:
        """
        Adjudicate an ambiguous field classification.
        
        Args:
            candidate: The field candidate
            classification: LayoutLM classification result
            page_image: Optional PIL Image of the page
        
        Returns:
            AdjudicationResult with final decision
        """
        import time
        start_time = time.time()
        
        # Generate input hash for reproducibility
        input_hash = self._hash_input(candidate, classification)
        
        if not self._api_available:
            # Fallback: use heuristic adjudication
            result = self._adjudicate_heuristic(candidate, classification)
            result.input_hash = input_hash
            result.processing_time_ms = int((time.time() - start_time) * 1000)
            self._log_adjudication(result)
            return result
        
        try:
            result = self._adjudicate_with_glm(candidate, classification, page_image)
            result.input_hash = input_hash
            result.processing_time_ms = int((time.time() - start_time) * 1000)
            
        except Exception as e:
            logger.error(f"GLM adjudication failed: {e}. Using fallback.")
            result = self._adjudicate_heuristic(candidate, classification)
            result.input_hash = input_hash
            result.processing_time_ms = int((time.time() - start_time) * 1000)
        
        self._log_adjudication(result)
        return result
    
    def adjudicate_batch(
        self,
        candidates_and_classifications: List[Tuple[FieldCandidate, ClassificationResult]],
        page_image: Optional[Any] = None
    ) -> List[AdjudicationResult]:
        """
        Adjudicate multiple ambiguous fields.
        
        Note: Currently processes sequentially. Could be parallelized
        for production deployment.
        
        Args:
            candidates_and_classifications: List of (candidate, classification) tuples
            page_image: Optional PIL Image of the page
        
        Returns:
            List of AdjudicationResult objects
        """
        results = []
        for candidate, classification in candidates_and_classifications:
            result = self.adjudicate(candidate, classification, page_image)
            results.append(result)
        return results
    
    def _adjudicate_with_glm(
        self,
        candidate: FieldCandidate,
        classification: ClassificationResult,
        page_image: Optional[Any] = None
    ) -> AdjudicationResult:
        """
        Perform adjudication using GLM-4.5V API.
        
        Constructs a multimodal prompt with text and optional image.
        """
        import requests
        
        # Build prompt
        prompt = self._build_adjudication_prompt(candidate, classification)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        # Add image if available
        if page_image is not None:
            # Convert image to base64
            image_data = self._prepare_image(page_image, candidate.bounding_box)
            if image_data:
                messages[-1] = {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]
                }
        
        # Make API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,  # Deterministic output
            "max_tokens": 200
        }
        
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result_data = response.json()
        glm_response = result_data['choices'][0]['message']['content']
        tokens_used = result_data.get('usage', {}).get('total_tokens', 0)
        
        # Parse GLM response
        return self._parse_glm_response(
            candidate.candidate_id,
            classification,
            glm_response,
            tokens_used
        )
    
    def _adjudicate_heuristic(
        self,
        candidate: FieldCandidate,
        classification: ClassificationResult
    ) -> AdjudicationResult:
        """
        Heuristic fallback adjudication when GLM is unavailable.
        
        Uses rule-based logic to make decisions.
        """
        # If confidence is reasonably high, accept primary
        if classification.primary_confidence >= 0.5:
            return AdjudicationResult(
                candidate_id=candidate.candidate_id,
                original_label=classification.primary_label,
                original_confidence=classification.primary_confidence,
                decision=AdjudicationDecision.ACCEPT_PRIMARY,
                final_label=classification.primary_label,
                final_confidence=classification.primary_confidence - 0.05,  # Slight penalty for uncertainty
                justification="Heuristic: Primary label accepted with moderate confidence",
                confidence_adjustment=-0.05,
                model_version="heuristic_fallback"
            )
        
        # If there's a strong alternative, consider it
        if classification.alternatives:
            best_alt, alt_conf = classification.alternatives[0]
            if alt_conf > classification.primary_confidence * 0.8:
                # Alternative is competitive
                return AdjudicationResult(
                    candidate_id=candidate.candidate_id,
                    original_label=classification.primary_label,
                    original_confidence=classification.primary_confidence,
                    decision=AdjudicationDecision.ACCEPT_ALTERNATIVE,
                    final_label=best_alt,
                    final_confidence=alt_conf,
                    justification=f"Heuristic: Alternative label {best_alt.value} preferred due to competitive confidence",
                    confidence_adjustment=alt_conf - classification.primary_confidence,
                    model_version="heuristic_fallback"
                )
        
        # Low confidence and no good alternative - mark unknown
        return AdjudicationResult(
            candidate_id=candidate.candidate_id,
            original_label=classification.primary_label,
            original_confidence=classification.primary_confidence,
            decision=AdjudicationDecision.MARK_UNKNOWN,
            final_label=SemanticLabel.UNKNOWN_FIELD,
            final_confidence=0.4,
            justification="Heuristic: Insufficient confidence to determine label",
            confidence_adjustment=0.4 - classification.primary_confidence,
            model_version="heuristic_fallback"
        )
    
    def _build_adjudication_prompt(
        self,
        candidate: FieldCandidate,
        classification: ClassificationResult
    ) -> str:
        """Build the adjudication prompt from template."""
        # Format alternatives
        alt_str = ", ".join([
            f"{label.value} ({conf:.2f})"
            for label, conf in classification.alternatives[:5]
        ]) if classification.alternatives else "None"
        
        # Get relevant ontology subset (to avoid overwhelming context)
        relevant_labels = self._get_relevant_labels(candidate, classification)
        
        return self.ADJUDICATION_PROMPT_TEMPLATE.format(
            field_type=candidate.field_type.value,
            x1=candidate.bounding_box.left,
            y1=candidate.bounding_box.top,
            x2=candidate.bounding_box.right,
            y2=candidate.bounding_box.bottom,
            detection_confidence=candidate.detection_confidence,
            context_text=candidate.get_context_text(max_chars=400),
            supporting_text=candidate.supporting_text or "(none detected)",
            primary_label=classification.primary_label.value,
            primary_confidence=classification.primary_confidence,
            alternatives=alt_str,
            ambiguity_reason=classification.ambiguity_reason or "Confidence below threshold",
            ontology_labels=", ".join(relevant_labels)
        )
    
    def _get_relevant_labels(
        self,
        candidate: FieldCandidate,
        classification: ClassificationResult
    ) -> List[str]:
        """
        Get a relevant subset of ontology labels for the prompt.
        
        Reduces context size while keeping useful options.
        """
        from .field_candidates import FieldType
        
        # Always include primary and alternatives
        relevant = {classification.primary_label.value}
        for label, _ in classification.alternatives:
            relevant.add(label.value)
        
        # Add field-type-specific labels
        if candidate.field_type == FieldType.CHECKBOX:
            checkbox_labels = [l.value for l in SemanticLabel if "CHECKBOX" in l.value]
            relevant.update(checkbox_labels)
            relevant.update([
                "AGE_OVER_18_CONFIRMATION",
                "UNDER_PENALTY_OF_PERJURY",
                "DECLARATION_CONFIRMATION"
            ])
        elif candidate.field_type == FieldType.LINE:
            # Add common line field labels
            relevant.update([
                "PERSON_NAME", "DATE", "SIGNATURE", "ADDRESS", "PHONE", "EMAIL",
                "CASE_NUMBER", "COURT_NAME", "ATTORNEY_NAME", "AMOUNT"
            ])
        
        # Always include unknown
        relevant.add("UNKNOWN_FIELD")
        
        return sorted(list(relevant))
    
    def _parse_glm_response(
        self,
        candidate_id: str,
        classification: ClassificationResult,
        glm_response: str,
        tokens_used: int
    ) -> AdjudicationResult:
        """
        Parse GLM response and validate against ontology.
        
        Handles malformed responses gracefully.
        """
        try:
            # Try to extract JSON from response
            response_text = glm_response.strip()
            
            # Handle potential markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            data = json.loads(response_text)
            
            selected_label_str = data.get("selected_label", "UNKNOWN_FIELD")
            justification = data.get("justification", "")
            confidence_adj = float(data.get("confidence_adjustment", 0.0))
            
            # Validate label is in ontology (CRITICAL)
            if not self.ontology.is_valid_label(selected_label_str):
                logger.warning(
                    f"GLM suggested invalid label '{selected_label_str}'. "
                    "Using UNKNOWN_FIELD instead."
                )
                selected_label_str = "UNKNOWN_FIELD"
                justification = f"Invalid label suggested: {data.get('selected_label')}. " + justification
            
            selected_label = SemanticLabel(selected_label_str)
            
            # Determine decision
            if selected_label == classification.primary_label:
                decision = AdjudicationDecision.ACCEPT_PRIMARY
            elif selected_label == SemanticLabel.UNKNOWN_FIELD:
                decision = AdjudicationDecision.MARK_UNKNOWN
            else:
                decision = AdjudicationDecision.ACCEPT_ALTERNATIVE
            
            # Calculate final confidence
            final_confidence = min(
                max(classification.primary_confidence + confidence_adj, 0.0),
                1.0
            )
            
            return AdjudicationResult(
                candidate_id=candidate_id,
                original_label=classification.primary_label,
                original_confidence=classification.primary_confidence,
                decision=decision,
                final_label=selected_label,
                final_confidence=final_confidence,
                glm_response=glm_response,
                justification=justification,
                confidence_adjustment=confidence_adj,
                tokens_used=tokens_used
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to parse GLM response: {e}. Response: {glm_response}")
            
            # Return conservative result
            return AdjudicationResult(
                candidate_id=candidate_id,
                original_label=classification.primary_label,
                original_confidence=classification.primary_confidence,
                decision=AdjudicationDecision.ACCEPT_PRIMARY,
                final_label=classification.primary_label,
                final_confidence=classification.primary_confidence - 0.1,
                glm_response=glm_response,
                justification=f"Parse error: {str(e)}. Keeping original label.",
                confidence_adjustment=-0.1,
                tokens_used=tokens_used
            )
    
    def _prepare_image(
        self,
        page_image: Any,
        bbox: BoundingBox,
        context_padding: float = 0.1
    ) -> Optional[str]:
        """
        Prepare image crop around field for GLM input.
        
        Adds bounding box overlay and converts to base64.
        """
        try:
            from PIL import Image, ImageDraw
            import base64
            from io import BytesIO
            
            # Get image dimensions
            width, height = page_image.size
            
            # Calculate crop region with padding
            pad_x = int(width * context_padding)
            pad_y = int(height * context_padding)
            
            left = max(0, int(bbox.left * width) - pad_x)
            top = max(0, int(bbox.top * height) - pad_y)
            right = min(width, int(bbox.right * width) + pad_x)
            bottom = min(height, int(bbox.bottom * height) + pad_y)
            
            # Crop image
            cropped = page_image.crop((left, top, right, bottom))
            
            # Draw bounding box overlay
            draw = ImageDraw.Draw(cropped)
            overlay_left = int(bbox.left * width) - left
            overlay_top = int(bbox.top * height) - top
            overlay_right = int(bbox.right * width) - left
            overlay_bottom = int(bbox.bottom * height) - top
            
            draw.rectangle(
                [overlay_left, overlay_top, overlay_right, overlay_bottom],
                outline="red",
                width=2
            )
            
            # Convert to base64
            buffer = BytesIO()
            cropped.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Failed to prepare image: {e}")
            return None
    
    def _hash_input(
        self,
        candidate: FieldCandidate,
        classification: ClassificationResult
    ) -> str:
        """Generate hash of input for reproducibility verification."""
        input_data = {
            'candidate_id': candidate.candidate_id,
            'field_type': candidate.field_type.value,
            'bbox': candidate.bounding_box.to_list(),
            'supporting_text': candidate.supporting_text,
            'context': candidate.get_context_text(max_chars=200),
            'primary_label': classification.primary_label.value,
            'primary_confidence': round(classification.primary_confidence, 4)
        }
        return hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()[:16]
    
    def _log_adjudication(self, result: AdjudicationResult):
        """Log adjudication result for audit trail."""
        self._adjudication_log.append(result)
        
        if self.enable_logging:
            logger.info(
                f"Adjudication [{result.candidate_id}]: "
                f"{result.original_label.value} ({result.original_confidence:.2f}) -> "
                f"{result.final_label.value} ({result.final_confidence:.2f}) | "
                f"Decision: {result.decision.value}"
            )
    
    def get_audit_log(self) -> List[Dict[str, Any]]:
        """
        Get full adjudication audit log.
        
        Returns list of dictionaries suitable for JSON serialization.
        """
        return [
            {
                'candidate_id': r.candidate_id,
                'original_label': r.original_label.value,
                'original_confidence': r.original_confidence,
                'final_label': r.final_label.value,
                'final_confidence': r.final_confidence,
                'decision': r.decision.value,
                'justification': r.justification,
                'timestamp': r.timestamp,
                'input_hash': r.input_hash,
                'model_version': r.model_version,
                'processing_time_ms': r.processing_time_ms,
                'tokens_used': r.tokens_used
            }
            for r in self._adjudication_log
        ]
    
    def clear_audit_log(self):
        """Clear the adjudication audit log."""
        self._adjudication_log = []

