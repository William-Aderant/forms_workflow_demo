"""
Technology Stack Comparison Pipeline
=====================================

Processes the same form through different technology configurations to
demonstrate the incremental value of each component in the stack.

Comparison Modes:
-----------------
1. TEXTRACT_ONLY: Just Textract OCR + geometry (baseline)
2. TEXTRACT_HEURISTICS: Textract + rule-based field classification
3. TEXTRACT_LAYOUTLM: Textract + LayoutLMv3 classification
4. FULL_STACK: Textract + LayoutLMv3 + GLM-4.5V adjudication

This allows side-by-side comparison to show:
- What each technology contributes
- Where the VLM helps with ambiguous cases
- Accuracy improvements at each stage

Image Comparison:
-----------------
Each mode generates an annotated image showing:
- Bounding boxes colored by confidence level
- Semantic labels overlaid on fields
- Visual progression from baseline to full stack
"""

import logging
import time
import base64
from io import BytesIO
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

from .ontology import SemanticLabel, ONTOLOGY
from .field_candidates import FieldCandidateDetector, FieldCandidate, FieldType, BoundingBox
from .layoutlm_classifier import LayoutLMClassifier, ClassificationResult
from .glm_adjudicator import GLMAdjudicator, AdjudicationResult
from .pipeline import FieldResult, PageResult

logger = logging.getLogger(__name__)


# ============================================================================
# Image Annotation Colors
# ============================================================================

# Color scheme for confidence levels (RGB tuples)
CONFIDENCE_COLORS = {
    'high': (76, 175, 80),      # Green - confidence >= 0.7
    'medium': (255, 152, 0),    # Orange - confidence 0.4-0.7
    'low': (244, 67, 54),       # Red - confidence < 0.4
    'unknown': (158, 158, 158), # Grey - UNKNOWN_FIELD
}

# Color scheme for field types
FIELD_TYPE_COLORS = {
    'line': (33, 150, 243),     # Blue
    'checkbox': (156, 39, 176), # Purple
    'table_cell': (0, 150, 136), # Teal
}

# Mode-specific accent colors for headers
MODE_COLORS = {
    'textract_only': (244, 67, 54),      # Red
    'textract_heuristics': (255, 152, 0), # Orange
    'textract_layoutlm': (33, 150, 243),  # Blue
    'full_stack': (76, 175, 80),          # Green
}


class ComparisonMode(str, Enum):
    """Technology stack configurations for comparison."""
    TEXTRACT_ONLY = "textract_only"
    TEXTRACT_HEURISTICS = "textract_heuristics"
    TEXTRACT_LAYOUTLM = "textract_layoutlm"
    FULL_STACK = "full_stack"


@dataclass
class ModeResult:
    """Results from a single processing mode."""
    mode: str
    mode_description: str
    fields: List[FieldResult]
    processing_time_ms: int
    
    # Mode-specific metadata
    components_used: List[str]
    
    # Annotated image (base64 encoded PNG)
    annotated_image_base64: Optional[str] = None
    
    # Statistics
    total_fields: int = 0
    high_confidence_count: int = 0  # >= 0.7
    medium_confidence_count: int = 0  # 0.4 - 0.7
    low_confidence_count: int = 0  # < 0.4
    unknown_field_count: int = 0
    classified_field_count: int = 0  # Non-UNKNOWN fields
    average_confidence: float = 0.0
    effective_confidence: float = 0.0  # Weighted confidence penalizing unknowns
    classification_rate: float = 0.0  # Ratio of classified vs unknown
    
    def __post_init__(self):
        """Calculate statistics after initialization."""
        self.total_fields = len(self.fields)
        
        conf_sum = 0.0
        weighted_conf_sum = 0.0
        
        for f in self.fields:
            conf_sum += f.confidence
            
            if f.confidence >= 0.7:
                self.high_confidence_count += 1
            elif f.confidence >= 0.4:
                self.medium_confidence_count += 1
            else:
                self.low_confidence_count += 1
            
            if f.semantic_label == "UNKNOWN_FIELD":
                self.unknown_field_count += 1
                # Penalize unknown fields - they contribute 0 to effective confidence
                weighted_conf_sum += 0.0
            else:
                self.classified_field_count += 1
                # Classified fields contribute their full confidence
                weighted_conf_sum += f.confidence
        
        # Raw average confidence (all fields)
        self.average_confidence = conf_sum / len(self.fields) if self.fields else 0.0
        
        # Effective confidence: weighted average that penalizes unknowns
        # This better reflects actual utility - UNKNOWN fields are essentially useless
        self.effective_confidence = weighted_conf_sum / len(self.fields) if self.fields else 0.0
        
        # Classification rate: what percentage of fields got meaningful labels
        self.classification_rate = self.classified_field_count / len(self.fields) if self.fields else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mode': self.mode,
            'mode_description': self.mode_description,
            'components_used': self.components_used,
            'fields': [f.to_extended_dict() for f in self.fields],
            'processing_time_ms': self.processing_time_ms,
            'annotated_image': self.annotated_image_base64,
            'statistics': {
                'total_fields': self.total_fields,
                'classified_field_count': self.classified_field_count,
                'unknown_field_count': self.unknown_field_count,
                'high_confidence_count': self.high_confidence_count,
                'medium_confidence_count': self.medium_confidence_count,
                'low_confidence_count': self.low_confidence_count,
                'average_confidence': round(self.average_confidence, 4),
                'effective_confidence': round(self.effective_confidence, 4),
                'classification_rate': round(self.classification_rate, 4)
            }
        }


@dataclass
class FieldComparison:
    """Comparison of a single field across all modes."""
    field_id: str
    field_type: str
    bounding_box: List[float]
    supporting_text: str
    
    # Results from each mode
    results_by_mode: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Analysis
    label_changed: bool = False
    confidence_improved: bool = False
    best_mode: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'field_id': self.field_id,
            'field_type': self.field_type,
            'bounding_box': self.bounding_box,
            'supporting_text': self.supporting_text,
            'results_by_mode': self.results_by_mode,
            'analysis': {
                'label_changed': self.label_changed,
                'confidence_improved': self.confidence_improved,
                'best_mode': self.best_mode
            }
        }


@dataclass
class ComparisonOutput:
    """Complete comparison output for a document."""
    document_id: str
    page_number: int
    
    # Results from each mode
    mode_results: Dict[str, ModeResult]
    
    # Field-by-field comparison
    field_comparisons: List[FieldComparison]
    
    # Overall analysis
    total_processing_time_ms: int = 0
    improvement_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'document_id': self.document_id,
            'page_number': self.page_number,
            'mode_results': {k: v.to_dict() for k, v in self.mode_results.items()},
            'field_comparisons': [fc.to_dict() for fc in self.field_comparisons],
            'total_processing_time_ms': self.total_processing_time_ms,
            'improvement_summary': self.improvement_summary,
            'timestamp': self.timestamp
        }


class ComparisonPipeline:
    """
    Pipeline that processes forms through multiple technology configurations
    for side-by-side comparison.
    
    Usage:
        pipeline = ComparisonPipeline()
        result = pipeline.compare(textract_blocks, page_image)
        
        # View results by mode
        for mode, mode_result in result.mode_results.items():
            print(f"{mode}: {mode_result.average_confidence:.2%} avg confidence")
        
        # View field-by-field comparison
        for field_comp in result.field_comparisons:
            print(f"Field {field_comp.field_id}: improved = {field_comp.confidence_improved}")
    """
    
    MODE_DESCRIPTIONS = {
        ComparisonMode.TEXTRACT_ONLY: "Textract OCR only - detects fields but no semantic classification",
        ComparisonMode.TEXTRACT_HEURISTICS: "Textract + rule-based pattern matching for field labels",
        ComparisonMode.TEXTRACT_LAYOUTLM: "Textract + LayoutLMv3 layout-aware classification",
        ComparisonMode.FULL_STACK: "Full stack: Textract + LayoutLMv3 + GLM-4.5V adjudication"
    }
    
    MODE_COMPONENTS = {
        ComparisonMode.TEXTRACT_ONLY: ["AWS Textract"],
        ComparisonMode.TEXTRACT_HEURISTICS: ["AWS Textract", "Rule-based Heuristics"],
        ComparisonMode.TEXTRACT_LAYOUTLM: ["AWS Textract", "LayoutLMv3"],
        ComparisonMode.FULL_STACK: ["AWS Textract", "LayoutLMv3", "GLM-4.5V"]
    }
    
    def __init__(
        self,
        layoutlm_model: str = "microsoft/layoutlmv3-base",
        use_gpu: bool = True,
        glm_api_key: Optional[str] = None
    ):
        """
        Initialize comparison pipeline.
        
        Args:
            layoutlm_model: HuggingFace model name for LayoutLM
            use_gpu: Whether to use GPU for LayoutLM
            glm_api_key: API key for GLM service
        """
        self.field_detector = FieldCandidateDetector()
        self.ontology = ONTOLOGY
        
        # Initialize LayoutLM classifier
        self.layoutlm_classifier = LayoutLMClassifier(
            model_name=layoutlm_model,
            use_gpu=use_gpu,
            fallback_to_heuristics=True
        )
        
        # Initialize GLM adjudicator
        self.glm_adjudicator = GLMAdjudicator(
            api_key=glm_api_key,
            enable_logging=True
        )
        
        logger.info(
            f"Initialized ComparisonPipeline - "
            f"LayoutLM mode: {self.layoutlm_classifier.mode}, "
            f"GLM available: {self.glm_adjudicator.is_available}"
        )
    
    def compare(
        self,
        textract_blocks: List[Dict[str, Any]],
        page_image: Optional[Any] = None,
        page_number: int = 1,
        document_id: str = "comparison",
        generate_images: bool = True
    ) -> ComparisonOutput:
        """
        Process a page through all technology configurations.
        
        Args:
            textract_blocks: Raw Textract blocks
            page_image: Optional PIL Image of the page
            page_number: Page number (1-indexed)
            document_id: Document identifier
            generate_images: Whether to generate annotated comparison images
        
        Returns:
            ComparisonOutput with results from all modes
        """
        start_time = time.time()
        
        logger.info(f"Starting comparison pipeline for document {document_id}")
        
        # Detect field candidates (shared across all modes)
        candidates = self.field_detector.detect_candidates(
            textract_blocks=textract_blocks,
            page_number=page_number
        )
        
        logger.info(f"Detected {len(candidates)} field candidates")
        
        if not candidates:
            return ComparisonOutput(
                document_id=document_id,
                page_number=page_number,
                mode_results={},
                field_comparisons=[],
                total_processing_time_ms=int((time.time() - start_time) * 1000)
            )
        
        # Process through each mode
        mode_results = {}
        
        # Mode 1: TEXTRACT_ONLY
        mode_results[ComparisonMode.TEXTRACT_ONLY.value] = self._process_textract_only(
            candidates, page_number
        )
        
        # Mode 2: TEXTRACT_HEURISTICS
        mode_results[ComparisonMode.TEXTRACT_HEURISTICS.value] = self._process_with_heuristics(
            candidates, page_number
        )
        
        # Mode 3: TEXTRACT_LAYOUTLM
        mode_results[ComparisonMode.TEXTRACT_LAYOUTLM.value] = self._process_with_layoutlm(
            candidates, page_image, page_number
        )
        
        # Mode 4: FULL_STACK
        mode_results[ComparisonMode.FULL_STACK.value] = self._process_full_stack(
            candidates, page_image, page_number
        )
        
        # Generate annotated images for each mode
        if generate_images and page_image is not None:
            logger.info("Generating annotated comparison images")
            for mode_name, mode_result in mode_results.items():
                try:
                    annotated_image = self._generate_annotated_image(
                        page_image=page_image,
                        mode_result=mode_result,
                        mode_name=mode_name
                    )
                    mode_result.annotated_image_base64 = annotated_image
                except Exception as e:
                    logger.warning(f"Failed to generate image for {mode_name}: {e}")
        
        # Build field-by-field comparisons
        field_comparisons = self._build_field_comparisons(candidates, mode_results)
        
        # Calculate improvement summary
        improvement_summary = self._calculate_improvements(mode_results)
        
        total_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"Comparison complete in {total_time_ms}ms")
        
        return ComparisonOutput(
            document_id=document_id,
            page_number=page_number,
            mode_results=mode_results,
            field_comparisons=field_comparisons,
            total_processing_time_ms=total_time_ms,
            improvement_summary=improvement_summary
        )
    
    def _process_textract_only(
        self,
        candidates: List[FieldCandidate],
        page_number: int
    ) -> ModeResult:
        """
        Mode 1: Only Textract field detection, no classification.
        
        All fields get UNKNOWN_FIELD label since we're not classifying.
        """
        start_time = time.time()
        
        fields = []
        for candidate in candidates:
            fields.append(FieldResult(
                field_id=candidate.candidate_id,
                field_type=candidate.field_type.value,
                bounding_box=candidate.bounding_box.to_list(),
                semantic_label="UNKNOWN_FIELD",
                confidence=candidate.detection_confidence * 0.3,  # Low confidence without classification
                supporting_text=candidate.supporting_text,
                detection_method=candidate.detection_method,
                classification_method="none"
            ))
        
        return ModeResult(
            mode=ComparisonMode.TEXTRACT_ONLY.value,
            mode_description=self.MODE_DESCRIPTIONS[ComparisonMode.TEXTRACT_ONLY],
            fields=fields,
            processing_time_ms=int((time.time() - start_time) * 1000),
            components_used=self.MODE_COMPONENTS[ComparisonMode.TEXTRACT_ONLY]
        )
    
    def _process_with_heuristics(
        self,
        candidates: List[FieldCandidate],
        page_number: int
    ) -> ModeResult:
        """
        Mode 2: Textract + rule-based heuristic classification.
        
        Uses simple pattern matching without LayoutLM.
        """
        start_time = time.time()
        
        fields = []
        for candidate in candidates:
            # Use heuristic classification
            label, confidence, _ = self._heuristic_classify(
                candidate.supporting_text or "",
                candidate.get_context_text(max_chars=200),
                candidate
            )
            
            fields.append(FieldResult(
                field_id=candidate.candidate_id,
                field_type=candidate.field_type.value,
                bounding_box=candidate.bounding_box.to_list(),
                semantic_label=label.value,
                confidence=round(confidence, 4),
                supporting_text=candidate.supporting_text,
                detection_method=candidate.detection_method,
                classification_method="heuristic"
            ))
        
        return ModeResult(
            mode=ComparisonMode.TEXTRACT_HEURISTICS.value,
            mode_description=self.MODE_DESCRIPTIONS[ComparisonMode.TEXTRACT_HEURISTICS],
            fields=fields,
            processing_time_ms=int((time.time() - start_time) * 1000),
            components_used=self.MODE_COMPONENTS[ComparisonMode.TEXTRACT_HEURISTICS]
        )
    
    def _process_with_layoutlm(
        self,
        candidates: List[FieldCandidate],
        page_image: Optional[Any],
        page_number: int
    ) -> ModeResult:
        """
        Mode 3: Textract + LayoutLMv3 classification (no GLM adjudication).
        """
        start_time = time.time()
        
        # Classify with LayoutLM
        classifications = self.layoutlm_classifier.classify_candidates(
            candidates=candidates,
            page_image=page_image
        )
        
        # Build classification map
        classification_map = {c.candidate_id: c for c in classifications}
        
        fields = []
        for candidate in candidates:
            classification = classification_map.get(candidate.candidate_id)
            
            if classification:
                label = classification.primary_label
                confidence = classification.primary_confidence
                method = classification.method
            else:
                label = SemanticLabel.UNKNOWN_FIELD
                confidence = 0.3
                method = "fallback"
            
            # Apply unknown threshold
            if confidence < 0.4:
                label = SemanticLabel.UNKNOWN_FIELD
                confidence = min(confidence, 0.4)
            
            fields.append(FieldResult(
                field_id=candidate.candidate_id,
                field_type=candidate.field_type.value,
                bounding_box=candidate.bounding_box.to_list(),
                semantic_label=label.value,
                confidence=round(confidence, 4),
                supporting_text=candidate.supporting_text,
                detection_method=candidate.detection_method,
                classification_method=method
            ))
        
        return ModeResult(
            mode=ComparisonMode.TEXTRACT_LAYOUTLM.value,
            mode_description=self.MODE_DESCRIPTIONS[ComparisonMode.TEXTRACT_LAYOUTLM],
            fields=fields,
            processing_time_ms=int((time.time() - start_time) * 1000),
            components_used=self.MODE_COMPONENTS[ComparisonMode.TEXTRACT_LAYOUTLM]
        )
    
    def _process_full_stack(
        self,
        candidates: List[FieldCandidate],
        page_image: Optional[Any],
        page_number: int
    ) -> ModeResult:
        """
        Mode 4: Full stack - Textract + LayoutLMv3 + GLM adjudication.
        """
        start_time = time.time()
        
        # Classify with LayoutLM
        classifications = self.layoutlm_classifier.classify_candidates(
            candidates=candidates,
            page_image=page_image
        )
        
        classification_map = {c.candidate_id: c for c in classifications}
        
        # Identify ambiguous cases for GLM
        adjudication_map: Dict[str, AdjudicationResult] = {}
        
        if self.glm_adjudicator:
            ambiguous_pairs = []
            for candidate in candidates:
                classification = classification_map.get(candidate.candidate_id)
                if classification and self.glm_adjudicator.should_adjudicate(candidate, classification):
                    ambiguous_pairs.append((candidate, classification))
            
            if ambiguous_pairs:
                logger.info(f"Adjudicating {len(ambiguous_pairs)} ambiguous fields with GLM")
                adjudication_results = self.glm_adjudicator.adjudicate_batch(
                    candidates_and_classifications=ambiguous_pairs,
                    page_image=page_image
                )
                
                for result in adjudication_results:
                    adjudication_map[result.candidate_id] = result
        
        # Build final results
        fields = []
        for candidate in candidates:
            classification = classification_map.get(candidate.candidate_id)
            adjudication = adjudication_map.get(candidate.candidate_id)
            
            if adjudication:
                label = adjudication.final_label
                confidence = adjudication.final_confidence
                was_adjudicated = True
                method = "layoutlm+glm"
            elif classification:
                label = classification.primary_label
                confidence = classification.primary_confidence
                was_adjudicated = False
                method = classification.method
            else:
                label = SemanticLabel.UNKNOWN_FIELD
                confidence = 0.3
                was_adjudicated = False
                method = "fallback"
            
            # Apply unknown threshold
            if confidence < 0.4:
                label = SemanticLabel.UNKNOWN_FIELD
                confidence = min(confidence, 0.4)
            
            fields.append(FieldResult(
                field_id=candidate.candidate_id,
                field_type=candidate.field_type.value,
                bounding_box=candidate.bounding_box.to_list(),
                semantic_label=label.value,
                confidence=round(confidence, 4),
                supporting_text=candidate.supporting_text,
                detection_method=candidate.detection_method,
                classification_method=method,
                was_adjudicated=was_adjudicated
            ))
        
        return ModeResult(
            mode=ComparisonMode.FULL_STACK.value,
            mode_description=self.MODE_DESCRIPTIONS[ComparisonMode.FULL_STACK],
            fields=fields,
            processing_time_ms=int((time.time() - start_time) * 1000),
            components_used=self.MODE_COMPONENTS[ComparisonMode.FULL_STACK]
        )
    
    def _heuristic_classify(
        self,
        text: str,
        context: str,
        candidate: FieldCandidate
    ) -> Tuple[SemanticLabel, float, List]:
        """
        Pattern-based heuristic classification using the improved LayoutLM classifier's
        heuristics for consistency across all modes.
        
        This delegates to the LayoutLM classifier's heuristic method to ensure
        consistent classification across TEXTRACT_HEURISTICS and the fallback
        modes in TEXTRACT_LAYOUTLM and FULL_STACK.
        """
        # Use the LayoutLM classifier's comprehensive heuristic classification
        return self.layoutlm_classifier._heuristic_classify(
            text.lower().strip(),
            context.lower(),
            candidate
        )
    
    def _build_field_comparisons(
        self,
        candidates: List[FieldCandidate],
        mode_results: Dict[str, ModeResult]
    ) -> List[FieldComparison]:
        """Build field-by-field comparison across all modes."""
        comparisons = []
        
        for candidate in candidates:
            results_by_mode = {}
            
            for mode_name, mode_result in mode_results.items():
                # Find this field in the mode results
                field_data = None
                for field in mode_result.fields:
                    if field.field_id == candidate.candidate_id:
                        field_data = {
                            'semantic_label': field.semantic_label,
                            'confidence': field.confidence,
                            'classification_method': field.classification_method,
                            'was_adjudicated': field.was_adjudicated
                        }
                        break
                
                if field_data:
                    results_by_mode[mode_name] = field_data
            
            # Analyze changes across modes
            labels = [r['semantic_label'] for r in results_by_mode.values()]
            confidences = [r['confidence'] for r in results_by_mode.values()]
            
            label_changed = len(set(labels)) > 1
            
            # Check if confidence improved from first to last mode
            confidence_improved = False
            if len(confidences) >= 2:
                confidence_improved = confidences[-1] > confidences[0]
            
            # Determine best mode (highest confidence for non-UNKNOWN label)
            best_mode = ""
            best_conf = -1
            for mode_name, data in results_by_mode.items():
                if data['semantic_label'] != "UNKNOWN_FIELD":
                    if data['confidence'] > best_conf:
                        best_conf = data['confidence']
                        best_mode = mode_name
            
            if not best_mode and results_by_mode:
                # All unknown, pick highest confidence
                best_mode = max(results_by_mode.keys(), 
                               key=lambda m: results_by_mode[m]['confidence'])
            
            comparisons.append(FieldComparison(
                field_id=candidate.candidate_id,
                field_type=candidate.field_type.value,
                bounding_box=candidate.bounding_box.to_list(),
                supporting_text=candidate.supporting_text,
                results_by_mode=results_by_mode,
                label_changed=label_changed,
                confidence_improved=confidence_improved,
                best_mode=best_mode
            ))
        
        return comparisons
    
    def _calculate_improvements(
        self,
        mode_results: Dict[str, ModeResult]
    ) -> Dict[str, Any]:
        """
        Calculate improvement metrics across modes.
        
        Uses effective_confidence (which penalizes UNKNOWN_FIELD) for more
        meaningful comparisons. Raw average_confidence can be misleading since
        UNKNOWN fields still contribute their confidence scores.
        """
        modes_ordered = [
            ComparisonMode.TEXTRACT_ONLY.value,
            ComparisonMode.TEXTRACT_HEURISTICS.value,
            ComparisonMode.TEXTRACT_LAYOUTLM.value,
            ComparisonMode.FULL_STACK.value
        ]
        
        improvements = {
            'confidence_progression': [],
            'classification_progression': [],
            'unknown_reduction': [],
            'high_confidence_increase': [],
            'mode_comparison': {}
        }
        
        prev_result = None
        for mode in modes_ordered:
            if mode not in mode_results:
                continue
            
            result = mode_results[mode]
            
            # Confidence progression - use effective confidence for primary metric
            improvements['confidence_progression'].append({
                'mode': mode,
                'average_confidence': round(result.average_confidence, 4),
                'effective_confidence': round(result.effective_confidence, 4)
            })
            
            # Classification rate progression - key metric for value demonstration
            improvements['classification_progression'].append({
                'mode': mode,
                'classified_count': result.classified_field_count,
                'classification_rate': round(result.classification_rate, 4)
            })
            
            improvements['unknown_reduction'].append({
                'mode': mode,
                'unknown_count': result.unknown_field_count,
                'unknown_rate': round(1.0 - result.classification_rate, 4)
            })
            
            improvements['high_confidence_increase'].append({
                'mode': mode,
                'high_confidence_count': result.high_confidence_count,
                'high_confidence_rate': round(result.high_confidence_count / result.total_fields, 4) if result.total_fields > 0 else 0
            })
            
            if prev_result:
                # Use effective confidence for mode-to-mode comparison
                conf_change = result.effective_confidence - prev_result.effective_confidence
                class_change = result.classified_field_count - prev_result.classified_field_count
                
                improvements['mode_comparison'][f"{prev_result.mode}_to_{mode}"] = {
                    'effective_confidence_change': round(conf_change, 4),
                    'raw_confidence_change': round(result.average_confidence - prev_result.average_confidence, 4),
                    'fields_classified_change': class_change,
                    'unknown_change': result.unknown_field_count - prev_result.unknown_field_count,
                    'high_confidence_change': result.high_confidence_count - prev_result.high_confidence_count,
                    'time_added_ms': result.processing_time_ms
                }
            
            prev_result = result
        
        # Overall improvement from baseline to full stack
        if ComparisonMode.TEXTRACT_ONLY.value in mode_results and ComparisonMode.FULL_STACK.value in mode_results:
            baseline = mode_results[ComparisonMode.TEXTRACT_ONLY.value]
            full = mode_results[ComparisonMode.FULL_STACK.value]
            
            # Calculate effective confidence gain
            eff_conf_gain = full.effective_confidence - baseline.effective_confidence
            eff_conf_gain_pct = (eff_conf_gain / max(baseline.effective_confidence, 0.01)) * 100 if baseline.effective_confidence > 0 else 0
            
            # Calculate raw confidence gain for comparison
            raw_conf_gain = full.average_confidence - baseline.average_confidence
            raw_conf_gain_pct = (raw_conf_gain / max(baseline.average_confidence, 0.01)) * 100 if baseline.average_confidence > 0 else 0
            
            improvements['overall_improvement'] = {
                # Primary metrics - based on effective confidence
                'effective_confidence_gain': round(eff_conf_gain, 4),
                'effective_confidence_gain_percent': round(eff_conf_gain_pct, 1),
                
                # Raw metrics for reference
                'raw_confidence_gain': round(raw_conf_gain, 4),
                'raw_confidence_gain_percent': round(raw_conf_gain_pct, 1),
                
                # Classification metrics - often more meaningful than confidence
                'fields_classified_gain': full.classified_field_count - baseline.classified_field_count,
                'classification_rate_gain': round(full.classification_rate - baseline.classification_rate, 4),
                
                # Legacy metrics
                'unknown_fields_reduced': baseline.unknown_field_count - full.unknown_field_count,
                'high_confidence_gained': full.high_confidence_count - baseline.high_confidence_count,
                
                # Processing cost
                'total_processing_time_ms': sum(r.processing_time_ms for r in mode_results.values())
            }
        
        # Find the best performing mode for each metric
        if mode_results:
            best_effective_conf_mode = max(mode_results.keys(), 
                                          key=lambda m: mode_results[m].effective_confidence)
            best_classification_mode = max(mode_results.keys(),
                                          key=lambda m: mode_results[m].classification_rate)
            best_high_conf_mode = max(mode_results.keys(),
                                     key=lambda m: mode_results[m].high_confidence_count)
            
            improvements['best_modes'] = {
                'best_effective_confidence': {
                    'mode': best_effective_conf_mode,
                    'value': round(mode_results[best_effective_conf_mode].effective_confidence, 4)
                },
                'best_classification_rate': {
                    'mode': best_classification_mode,
                    'value': round(mode_results[best_classification_mode].classification_rate, 4)
                },
                'most_high_confidence_fields': {
                    'mode': best_high_conf_mode,
                    'value': mode_results[best_high_conf_mode].high_confidence_count
                }
            }
        
        return improvements
    
    # =========================================================================
    # Image Generation Methods
    # =========================================================================
    
    def _generate_annotated_image(
        self,
        page_image: Any,
        mode_result: ModeResult,
        mode_name: str
    ) -> Optional[str]:
        """
        Generate an annotated image showing field detections for a processing mode.
        
        Args:
            page_image: PIL Image of the page
            mode_result: Results from processing mode
            mode_name: Name of the mode (for header)
        
        Returns:
            Base64 encoded PNG image string
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            logger.warning("PIL not available for image generation")
            return None
        
        # Header height for mode banner
        header_height = 40
        
        # Get original image dimensions
        orig_img = page_image.copy()
        if orig_img.mode != 'RGB':
            orig_img = orig_img.convert('RGB')
        orig_width, orig_height = orig_img.size
        
        # Create expanded canvas to include header (so bounding boxes stay aligned)
        img = Image.new('RGB', (orig_width, orig_height + header_height), (255, 255, 255))
        img.paste(orig_img, (0, header_height))  # Paste original image below header
        
        draw = ImageDraw.Draw(img)
        width = orig_width
        height = orig_height  # Use original height for bbox calculations
        
        # Try to load a font, fall back to default
        try:
            # Try common font paths
            font_paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/SFNSText.ttf",
                "C:\\Windows\\Fonts\\arial.ttf",
            ]
            font = None
            font_small = None
            font_header = None
            
            for fp in font_paths:
                try:
                    font = ImageFont.truetype(fp, 12)
                    font_small = ImageFont.truetype(fp, 10)
                    font_header = ImageFont.truetype(fp, 18)
                    break
                except (IOError, OSError):
                    continue
            
            if font is None:
                font = ImageFont.load_default()
                font_small = font
                font_header = font
        except Exception:
            font = ImageFont.load_default()
            font_small = font
            font_header = font
        
        # Draw header banner (header_height already defined above)
        mode_color = MODE_COLORS.get(mode_name, (100, 100, 100))
        draw.rectangle([0, 0, width, header_height], fill=mode_color)
        
        # Mode title
        mode_titles = {
            'textract_only': '1. TEXTRACT ONLY (Baseline)',
            'textract_heuristics': '2. TEXTRACT + HEURISTICS',
            'textract_layoutlm': '3. TEXTRACT + LAYOUTLM',
            'full_stack': '4. FULL STACK (Textract + LayoutLM + GLM)'
        }
        title = mode_titles.get(mode_name, mode_name.upper())
        draw.text((10, 10), title, fill=(255, 255, 255), font=font_header)
        
        # Stats on header
        stats_text = f"Fields: {mode_result.total_fields} | Avg Conf: {mode_result.average_confidence:.1%} | High Conf: {mode_result.high_confidence_count} | Unknown: {mode_result.unknown_field_count}"
        draw.text((width - 450, 12), stats_text, fill=(255, 255, 255), font=font_small)
        
        # Draw each field
        for field in mode_result.fields:
            self._draw_field_annotation(
                draw=draw,
                field=field,
                width=width,
                height=height,
                font=font,
                font_small=font_small,
                header_offset=header_height
            )
        
        # Draw legend (use full image height including header)
        self._draw_legend(draw, width, height + header_height, font_small)
        
        # Convert to base64
        buffer = BytesIO()
        img.save(buffer, format='PNG', quality=90)
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _draw_field_annotation(
        self,
        draw: Any,
        field: FieldResult,
        width: int,
        height: int,
        font: Any,
        font_small: Any,
        header_offset: int = 0
    ):
        """
        Draw annotation for a single field.
        
        Args:
            draw: PIL ImageDraw object
            field: Field result to annotate
            width: Image width
            height: Image height
            font: Font for labels
            font_small: Small font for confidence
            header_offset: Vertical offset for header
        """
        # Convert normalized bbox to pixel coordinates
        bbox = field.bounding_box
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height) + header_offset
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height) + header_offset
        
        # Determine color based on confidence and label
        if field.semantic_label == "UNKNOWN_FIELD":
            color = CONFIDENCE_COLORS['unknown']
        elif field.confidence >= 0.7:
            color = CONFIDENCE_COLORS['high']
        elif field.confidence >= 0.4:
            color = CONFIDENCE_COLORS['medium']
        else:
            color = CONFIDENCE_COLORS['low']
        
        # Draw bounding box with semi-transparent fill
        fill_color = (*color, 50)  # Add alpha for transparency
        
        # Since PIL doesn't support alpha directly in rectangle, draw outline
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw a thin colored bar at the top of the box
        bar_height = min(4, (y2 - y1) // 4)
        draw.rectangle([x1, y1, x2, y1 + bar_height], fill=color)
        
        # Format label text
        label_short = field.semantic_label.replace('_', ' ')
        if len(label_short) > 20:
            label_short = label_short[:18] + '..'
        
        # Confidence percentage
        conf_text = f"{field.confidence:.0%}"
        
        # Calculate label position (above box if room, else inside)
        label_y = y1 - 16 if y1 > header_offset + 20 else y1 + 2
        
        # Draw label background
        label_text = f"{label_short} ({conf_text})"
        try:
            text_bbox = draw.textbbox((x1, label_y), label_text, font=font_small)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        except AttributeError:
            # Fallback for older PIL versions
            text_width = len(label_text) * 6
            text_height = 12
        
        # Background rectangle for text
        padding = 2
        draw.rectangle(
            [x1 - padding, label_y - padding, x1 + text_width + padding, label_y + text_height + padding],
            fill=(255, 255, 255, 200)
        )
        
        # Draw the label text
        draw.text((x1, label_y), label_text, fill=color, font=font_small)
        
        # Add field type indicator
        type_colors = {
            'line': (33, 150, 243),
            'checkbox': (156, 39, 176),
            'table_cell': (0, 150, 136),
        }
        type_color = type_colors.get(field.field_type, (100, 100, 100))
        
        # Small indicator dot for field type
        indicator_radius = 4
        indicator_x = x2 - indicator_radius - 2
        indicator_y = y1 + indicator_radius + 2
        draw.ellipse(
            [indicator_x - indicator_radius, indicator_y - indicator_radius,
             indicator_x + indicator_radius, indicator_y + indicator_radius],
            fill=type_color
        )
    
    def _draw_legend(self, draw: Any, width: int, height: int, font: Any):
        """
        Draw a legend explaining the color coding.
        
        Args:
            draw: PIL ImageDraw object
            width: Image width
            height: Image height
            font: Font for legend text
        """
        legend_items = [
            ("High Conf (≥70%)", CONFIDENCE_COLORS['high']),
            ("Med Conf (40-70%)", CONFIDENCE_COLORS['medium']),
            ("Low Conf (<40%)", CONFIDENCE_COLORS['low']),
            ("Unknown Field", CONFIDENCE_COLORS['unknown']),
        ]
        
        # Legend position (bottom right)
        legend_width = 130
        legend_height = len(legend_items) * 18 + 10
        legend_x = width - legend_width - 10
        legend_y = height - legend_height - 10
        
        # Background
        draw.rectangle(
            [legend_x, legend_y, legend_x + legend_width, legend_y + legend_height],
            fill=(255, 255, 255, 230),
            outline=(200, 200, 200)
        )
        
        # Draw legend items
        y_offset = legend_y + 8
        for label, color in legend_items:
            # Color box
            draw.rectangle(
                [legend_x + 8, y_offset, legend_x + 20, y_offset + 12],
                fill=color
            )
            # Label
            draw.text((legend_x + 26, y_offset), label, fill=(50, 50, 50), font=font)
            y_offset += 18

