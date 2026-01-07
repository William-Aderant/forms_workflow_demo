"""
Legal Form Understanding Pipeline
=================================

The main orchestrator that coordinates all stages of the form understanding pipeline.

Pipeline Stages:
----------------
1. INPUT: Accept PDF, normalize DPI/orientation, split into pages
2. OCR + GEOMETRY: Extract with Textract (WORD, LINE, SELECTION_ELEMENT)
3. FIELD CANDIDATE DETECTION: Identify empty fields
4. SEMANTIC LABELING: LayoutLMv3 classification
5. AMBIGUITY HANDLING: GLM-4.5V adjudication
6. OUTPUT: Deterministic JSON schema

Design Principles:
------------------
- Geometry is ALWAYS from Textract (never modified)
- LayoutLM is the PRIMARY classifier
- GLM is ADVISORY only (for ambiguous cases)
- All outputs are deterministic
- Full audit trail for legal review

Output Schema (per pipeline spec):
----------------------------------
{
  "page": number,
  "fields": [
    {
      "field_id": string,
      "field_type": "line" | "checkbox" | "table_cell",
      "bounding_box": [x1, y1, x2, y2],
      "semantic_label": string,
      "confidence": float,
      "supporting_text": string
    }
  ]
}
"""

import logging
import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from .ontology import SemanticLabel, LegalFieldOntology, ONTOLOGY
from .field_candidates import FieldCandidateDetector, FieldCandidate, FieldType, BoundingBox
from .layoutlm_classifier import LayoutLMClassifier, ClassificationResult
from .glm_adjudicator import GLMAdjudicator, AdjudicationResult, AdjudicationDecision

logger = logging.getLogger(__name__)


@dataclass
class FieldResult:
    """
    Final result for a single detected field.
    
    Matches the output schema defined in the pipeline specification.
    """
    field_id: str
    field_type: str  # "line" | "checkbox" | "table_cell"
    bounding_box: List[float]  # [x1, y1, x2, y2]
    semantic_label: str
    confidence: float
    supporting_text: str
    
    # Extended metadata (for audit purposes)
    detection_method: str = ""
    classification_method: str = ""
    was_adjudicated: bool = False
    adjudication_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'field_id': self.field_id,
            'field_type': self.field_type,
            'bounding_box': self.bounding_box,
            'semantic_label': self.semantic_label,
            'confidence': self.confidence,
            'supporting_text': self.supporting_text
        }
    
    def to_extended_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with extended metadata."""
        return asdict(self)


@dataclass
class PageResult:
    """Results for a single page."""
    page: int
    fields: List[FieldResult]
    
    # Processing metadata
    processing_time_ms: int = 0
    textract_blocks_count: int = 0
    candidates_detected: int = 0
    candidates_classified: int = 0
    candidates_adjudicated: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization (minimal schema)."""
        return {
            'page': self.page,
            'fields': [f.to_dict() for f in self.fields]
        }
    
    def to_extended_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with extended metadata."""
        return {
            'page': self.page,
            'fields': [f.to_extended_dict() for f in self.fields],
            'processing_time_ms': self.processing_time_ms,
            'textract_blocks_count': self.textract_blocks_count,
            'candidates_detected': self.candidates_detected,
            'candidates_classified': self.candidates_classified,
            'candidates_adjudicated': self.candidates_adjudicated
        }


@dataclass
class PipelineOutput:
    """
    Complete pipeline output for a document.
    
    Contains results for all pages plus audit trail.
    """
    document_id: str
    pages: List[PageResult]
    
    # Document metadata
    total_pages: int = 0
    total_fields: int = 0
    total_adjudicated: int = 0
    
    # Processing metadata
    processing_start: str = ""
    processing_end: str = ""
    total_processing_time_ms: int = 0
    
    # Audit information
    pipeline_version: str = "1.0.0"
    input_hash: str = ""  # Hash of input for reproducibility
    
    # GLM adjudication audit log
    adjudication_log: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to minimal JSON schema as per spec."""
        return {
            'pages': [p.to_dict() for p in self.pages]
        }
    
    def to_extended_dict(self) -> Dict[str, Any]:
        """Convert to full JSON with all metadata."""
        return {
            'document_id': self.document_id,
            'pages': [p.to_extended_dict() for p in self.pages],
            'metadata': {
                'total_pages': self.total_pages,
                'total_fields': self.total_fields,
                'total_adjudicated': self.total_adjudicated,
                'processing_start': self.processing_start,
                'processing_end': self.processing_end,
                'total_processing_time_ms': self.total_processing_time_ms,
                'pipeline_version': self.pipeline_version,
                'input_hash': self.input_hash
            },
            'audit': {
                'adjudication_log': self.adjudication_log
            }
        }


class LegalFormPipeline:
    """
    Production-grade pipeline for legal form field detection and classification.
    
    Example usage:
    
        pipeline = LegalFormPipeline()
        
        # Process a PDF
        with open('form.pdf', 'rb') as f:
            pdf_bytes = f.read()
        
        result = pipeline.process_pdf(pdf_bytes)
        
        # Get minimal JSON output
        output = result.to_dict()
        
        # Get full audit trail
        full_output = result.to_extended_dict()
    """
    
    # Confidence threshold below which we mark as UNKNOWN
    UNKNOWN_CONFIDENCE_THRESHOLD = 0.4
    
    def __init__(
        self,
        textract_service=None,
        layoutlm_model: str = "microsoft/layoutlmv3-base",
        use_gpu: bool = True,
        enable_glm_adjudication: bool = True,
        glm_api_key: Optional[str] = None
    ):
        """
        Initialize the legal form pipeline.
        
        Args:
            textract_service: Optional pre-configured TextractService
            layoutlm_model: HuggingFace model name for LayoutLM
            use_gpu: Whether to use GPU for LayoutLM
            enable_glm_adjudication: Whether to use GLM for ambiguous cases
            glm_api_key: API key for GLM service
        """
        self.textract_service = textract_service
        self.enable_glm_adjudication = enable_glm_adjudication
        
        # Initialize components
        self.field_detector = FieldCandidateDetector()
        
        self.classifier = LayoutLMClassifier(
            model_name=layoutlm_model,
            use_gpu=use_gpu,
            fallback_to_heuristics=True
        )
        
        self.adjudicator = GLMAdjudicator(
            api_key=glm_api_key,
            enable_logging=True
        ) if enable_glm_adjudication else None
        
        self.ontology = ONTOLOGY
        
        logger.info(
            f"Initialized LegalFormPipeline - "
            f"LayoutLM mode: {self.classifier.mode}, "
            f"GLM adjudication: {enable_glm_adjudication}"
        )
    
    def process_pdf(
        self,
        pdf_bytes: bytes,
        document_id: Optional[str] = None
    ) -> PipelineOutput:
        """
        Process a PDF document through the full pipeline.
        
        Args:
            pdf_bytes: Raw PDF file bytes
            document_id: Optional document identifier
        
        Returns:
            PipelineOutput with all detected and classified fields
        """
        import time
        start_time = time.time()
        processing_start = datetime.utcnow().isoformat()
        
        # Generate document ID if not provided
        if not document_id:
            document_id = hashlib.sha256(pdf_bytes[:1000]).hexdigest()[:12]
        
        # Calculate input hash for reproducibility
        input_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]
        
        logger.info(f"Processing document {document_id} ({len(pdf_bytes)} bytes)")
        
        # STAGE 1: INPUT - Extract pages from PDF
        page_images = self._extract_pages(pdf_bytes)
        total_pages = len(page_images) if page_images else 1
        
        # STAGE 2: OCR + GEOMETRY - Process with Textract
        textract_results = self._run_textract(pdf_bytes)
        
        if not textract_results.get('success'):
            logger.error(f"Textract processing failed: {textract_results.get('error')}")
            return PipelineOutput(
                document_id=document_id,
                pages=[],
                total_pages=0,
                processing_start=processing_start,
                processing_end=datetime.utcnow().isoformat(),
                input_hash=input_hash
            )
        
        # Process each page
        page_results = []
        total_fields = 0
        total_adjudicated = 0
        
        # For now, process first page (multi-page support requires async Textract)
        # In production, would use StartDocumentTextDetection for multi-page
        for page_num in range(1, 2):  # Currently single page
            page_start = time.time()
            
            page_result = self._process_page(
                page_number=page_num,
                textract_blocks=textract_results.get('raw_blocks', []),
                page_image=page_images[0] if page_images else None
            )
            
            page_result.processing_time_ms = int((time.time() - page_start) * 1000)
            page_results.append(page_result)
            
            total_fields += len(page_result.fields)
            total_adjudicated += page_result.candidates_adjudicated
        
        # Build output
        processing_end = datetime.utcnow().isoformat()
        total_time_ms = int((time.time() - start_time) * 1000)
        
        output = PipelineOutput(
            document_id=document_id,
            pages=page_results,
            total_pages=total_pages,
            total_fields=total_fields,
            total_adjudicated=total_adjudicated,
            processing_start=processing_start,
            processing_end=processing_end,
            total_processing_time_ms=total_time_ms,
            input_hash=input_hash,
            adjudication_log=self.adjudicator.get_audit_log() if self.adjudicator else []
        )
        
        logger.info(
            f"Pipeline complete for {document_id}: "
            f"{total_fields} fields, {total_adjudicated} adjudicated, "
            f"{total_time_ms}ms total"
        )
        
        return output
    
    def process_textract_output(
        self,
        textract_blocks: List[Dict[str, Any]],
        page_image: Optional[Any] = None,
        page_number: int = 1,
        document_id: Optional[str] = None
    ) -> PipelineOutput:
        """
        Process pre-extracted Textract output.
        
        Useful when Textract has already been called separately.
        
        Args:
            textract_blocks: Raw Textract blocks
            page_image: Optional PIL Image of the page
            page_number: Page number (1-indexed)
            document_id: Optional document identifier
        
        Returns:
            PipelineOutput with detected and classified fields
        """
        import time
        start_time = time.time()
        processing_start = datetime.utcnow().isoformat()
        
        if not document_id:
            document_id = f"textract_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        page_result = self._process_page(
            page_number=page_number,
            textract_blocks=textract_blocks,
            page_image=page_image
        )
        
        page_result.processing_time_ms = int((time.time() - start_time) * 1000)
        
        output = PipelineOutput(
            document_id=document_id,
            pages=[page_result],
            total_pages=1,
            total_fields=len(page_result.fields),
            total_adjudicated=page_result.candidates_adjudicated,
            processing_start=processing_start,
            processing_end=datetime.utcnow().isoformat(),
            total_processing_time_ms=page_result.processing_time_ms,
            adjudication_log=self.adjudicator.get_audit_log() if self.adjudicator else []
        )
        
        return output
    
    def _extract_pages(self, pdf_bytes: bytes) -> Optional[List[Any]]:
        """
        Extract page images from PDF.
        
        Stage 1: INPUT normalization.
        """
        try:
            from pdf2image import convert_from_bytes
            
            images = convert_from_bytes(
                pdf_bytes,
                dpi=150,  # Balance quality vs processing time
                fmt='PNG'
            )
            
            logger.info(f"Extracted {len(images)} page(s) from PDF")
            return images
            
        except Exception as e:
            logger.warning(f"Failed to extract page images: {e}")
            return None
    
    def _run_textract(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Run Textract OCR on the document.
        
        Stage 2: OCR + GEOMETRY extraction.
        
        Returns raw blocks with geometry preserved exactly as Textract returns.
        If PDF format is not supported, converts to image and retries.
        """
        if self.textract_service is None:
            # Lazy import and initialize
            try:
                from app.services.textract_service import TextractService
                self.textract_service = TextractService()
            except Exception as e:
                logger.error(f"Failed to initialize TextractService: {e}")
                return {'success': False, 'error': str(e)}
        
        try:
            # Use analyze_document with FORMS for better field detection
            import boto3
            from botocore.exceptions import ClientError
            from app.config import Config
            from io import BytesIO
            
            config = Config.get_boto3_config()
            if 'profile_name' in config:
                session = boto3.Session(profile_name=config['profile_name'])
                client = session.client('textract', region_name=config['region_name'])
            else:
                client = boto3.client('textract', **config)
            
            # Check PDF size for sync vs async
            pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
            
            if pdf_size_mb > 5.0:
                logger.warning(
                    f"PDF size ({pdf_size_mb:.1f}MB) exceeds sync limit. "
                    "Consider using async Textract API."
                )
                return {
                    'success': False,
                    'error': 'PDF too large for synchronous processing'
                }
            
            # Try PDF directly first
            try:
                response = client.analyze_document(
                    Document={'Bytes': pdf_bytes},
                    FeatureTypes=['FORMS', 'TABLES']
                )
                
                return {
                    'success': True,
                    'raw_blocks': response.get('Blocks', []),
                    'metadata': response.get('DocumentMetadata', {})
                }
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == 'UnsupportedDocumentException':
                    # PDF format not supported, convert to image and retry
                    logger.warning("PDF format not supported by Textract, converting to image")
                    
                    try:
                        from pdf2image import convert_from_bytes
                        
                        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=150)
                        if not images:
                            return {
                                'success': False,
                                'error': 'Failed to convert PDF to image'
                            }
                        
                        # Convert to PNG bytes
                        buffer = BytesIO()
                        images[0].save(buffer, format='PNG')
                        image_bytes = buffer.getvalue()
                        logger.info(f"Converted PDF to image: {len(image_bytes)} bytes")
                        
                        # Retry with image
                        response = client.analyze_document(
                            Document={'Bytes': image_bytes},
                            FeatureTypes=['FORMS', 'TABLES']
                        )
                        
                        return {
                            'success': True,
                            'raw_blocks': response.get('Blocks', []),
                            'metadata': response.get('DocumentMetadata', {})
                        }
                    except Exception as conv_error:
                        logger.error(f"Failed to convert PDF and process: {conv_error}")
                        return {'success': False, 'error': str(conv_error)}
                else:
                    # Re-raise other ClientErrors
                    raise
                
        except Exception as e:
            logger.error(f"Textract processing failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _process_page(
        self,
        page_number: int,
        textract_blocks: List[Dict[str, Any]],
        page_image: Optional[Any] = None
    ) -> PageResult:
        """
        Process a single page through stages 3-5.
        
        Stage 3: FIELD CANDIDATE DETECTION
        Stage 4: SEMANTIC LABELING (LayoutLM)
        Stage 5: AMBIGUITY HANDLING (GLM)
        """
        logger.info(f"Processing page {page_number}")
        
        # === STAGE 3: FIELD CANDIDATE DETECTION ===
        candidates = self.field_detector.detect_candidates(
            textract_blocks=textract_blocks,
            page_number=page_number
        )
        
        logger.info(f"Page {page_number}: {len(candidates)} field candidates detected")
        
        if not candidates:
            return PageResult(
                page=page_number,
                fields=[],
                textract_blocks_count=len(textract_blocks),
                candidates_detected=0
            )
        
        # === STAGE 4: SEMANTIC LABELING (LayoutLM) ===
        classifications = self.classifier.classify_candidates(
            candidates=candidates,
            page_image=page_image
        )
        
        # Build candidate -> classification mapping
        classification_map = {c.candidate_id: c for c in classifications}
        
        # === STAGE 5: AMBIGUITY HANDLING (GLM) ===
        adjudicated_count = 0
        adjudication_map: Dict[str, AdjudicationResult] = {}
        
        if self.adjudicator and self.enable_glm_adjudication:
            # Identify candidates needing adjudication
            ambiguous_pairs = []
            for candidate in candidates:
                classification = classification_map.get(candidate.candidate_id)
                if classification and self.adjudicator.should_adjudicate(candidate, classification):
                    ambiguous_pairs.append((candidate, classification))
            
            if ambiguous_pairs:
                logger.info(f"Page {page_number}: {len(ambiguous_pairs)} fields need adjudication")
                
                adjudication_results = self.adjudicator.adjudicate_batch(
                    candidates_and_classifications=ambiguous_pairs,
                    page_image=page_image
                )
                
                for result in adjudication_results:
                    adjudication_map[result.candidate_id] = result
                    adjudicated_count += 1
        
        # === STAGE 6: BUILD FINAL OUTPUT ===
        field_results = []
        
        for candidate in candidates:
            classification = classification_map.get(candidate.candidate_id)
            adjudication = adjudication_map.get(candidate.candidate_id)
            
            # Determine final label and confidence
            if adjudication:
                final_label = adjudication.final_label
                final_confidence = adjudication.final_confidence
                was_adjudicated = True
                adjudication_reason = adjudication.justification
                classification_method = "layoutlm+glm"
            elif classification:
                final_label = classification.primary_label
                final_confidence = classification.primary_confidence
                was_adjudicated = False
                adjudication_reason = None
                classification_method = classification.method
            else:
                final_label = SemanticLabel.UNKNOWN_FIELD
                final_confidence = 0.3
                was_adjudicated = False
                adjudication_reason = None
                classification_method = "fallback"
            
            # Apply UNKNOWN threshold rule
            if final_confidence < self.UNKNOWN_CONFIDENCE_THRESHOLD:
                final_label = SemanticLabel.UNKNOWN_FIELD
                final_confidence = min(final_confidence, 0.4)
            
            field_results.append(FieldResult(
                field_id=candidate.candidate_id,
                field_type=candidate.field_type.value,
                bounding_box=candidate.bounding_box.to_list(),
                semantic_label=final_label.value,
                confidence=round(final_confidence, 4),
                supporting_text=candidate.supporting_text,
                detection_method=candidate.detection_method,
                classification_method=classification_method,
                was_adjudicated=was_adjudicated,
                adjudication_reason=adjudication_reason
            ))
        
        return PageResult(
            page=page_number,
            fields=field_results,
            textract_blocks_count=len(textract_blocks),
            candidates_detected=len(candidates),
            candidates_classified=len(classification_map),
            candidates_adjudicated=adjudicated_count
        )
    
    def get_statistics(self, output: PipelineOutput) -> Dict[str, Any]:
        """
        Generate statistics summary for a pipeline output.
        
        Useful for monitoring and quality assurance.
        """
        label_counts: Dict[str, int] = {}
        confidence_sum = 0.0
        field_type_counts: Dict[str, int] = {}
        
        for page in output.pages:
            for field in page.fields:
                # Label distribution
                label_counts[field.semantic_label] = label_counts.get(field.semantic_label, 0) + 1
                
                # Confidence stats
                confidence_sum += field.confidence
                
                # Field type distribution
                field_type_counts[field.field_type] = field_type_counts.get(field.field_type, 0) + 1
        
        avg_confidence = confidence_sum / output.total_fields if output.total_fields > 0 else 0.0
        
        return {
            'document_id': output.document_id,
            'total_pages': output.total_pages,
            'total_fields': output.total_fields,
            'total_adjudicated': output.total_adjudicated,
            'adjudication_rate': output.total_adjudicated / output.total_fields if output.total_fields > 0 else 0.0,
            'average_confidence': round(avg_confidence, 4),
            'label_distribution': label_counts,
            'field_type_distribution': field_type_counts,
            'unknown_field_count': label_counts.get('UNKNOWN_FIELD', 0),
            'unknown_field_rate': label_counts.get('UNKNOWN_FIELD', 0) / output.total_fields if output.total_fields > 0 else 0.0,
            'processing_time_ms': output.total_processing_time_ms
        }

