"""
Legal Form Pipeline API Routes
==============================

REST API endpoints for the legal form understanding pipeline.

Endpoints:
- POST /api/v1/legal-forms/analyze - Analyze a PDF for empty fields
- POST /api/v1/legal-forms/analyze-textract - Analyze pre-extracted Textract output
- POST /api/v1/legal-forms/compare - Compare results across technology configurations
- GET /api/v1/legal-forms/ontology - Get the semantic label ontology
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/legal-forms", tags=["Legal Form Pipeline"])


# ============================================================================
# Request/Response Models
# ============================================================================

class FieldOutput(BaseModel):
    """Single field result in output."""
    field_id: str
    field_type: str = Field(..., description="line | checkbox | table_cell")
    bounding_box: List[float] = Field(..., description="[x1, y1, x2, y2] normalized")
    semantic_label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_text: str


class PageOutput(BaseModel):
    """Single page result in output."""
    page: int
    fields: List[FieldOutput]


class AnalyzeResponse(BaseModel):
    """Response for analyze endpoint."""
    success: bool
    document_id: Optional[str] = None
    pages: List[PageOutput] = []
    statistics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ExtendedAnalyzeResponse(BaseModel):
    """Extended response with audit trail."""
    success: bool
    document_id: Optional[str] = None
    pages: List[Dict[str, Any]] = []
    metadata: Optional[Dict[str, Any]] = None
    audit: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TextractBlock(BaseModel):
    """Textract block input."""
    Id: str
    BlockType: str
    Text: Optional[str] = None
    Confidence: Optional[float] = None
    Geometry: Optional[Dict[str, Any]] = None
    Relationships: Optional[List[Dict[str, Any]]] = None
    EntityTypes: Optional[List[str]] = None
    SelectionStatus: Optional[str] = None
    RowIndex: Optional[int] = None
    ColumnIndex: Optional[int] = None


class TextractAnalyzeRequest(BaseModel):
    """Request for analyzing pre-extracted Textract output."""
    blocks: List[Dict[str, Any]]
    page_number: int = 1
    document_id: Optional[str] = None


class OntologyLabel(BaseModel):
    """Semantic label definition."""
    label: str
    aliases: List[str]
    typical_field_types: List[str]
    is_attestational: bool


class OntologyResponse(BaseModel):
    """Response containing the full ontology."""
    total_labels: int
    labels: List[OntologyLabel]


class ComparisonModeEnum(str, Enum):
    """Available comparison modes."""
    TEXTRACT_ONLY = "textract_only"
    TEXTRACT_HEURISTICS = "textract_heuristics"
    TEXTRACT_LAYOUTLM = "textract_layoutlm"
    FULL_STACK = "full_stack"


class ModeResultOutput(BaseModel):
    """Result from a single processing mode."""
    mode: str
    mode_description: str
    components_used: List[str]
    fields: List[Dict[str, Any]]
    processing_time_ms: int
    statistics: Dict[str, Any]
    annotated_image: Optional[str] = None  # Base64 encoded PNG


class FieldComparisonOutput(BaseModel):
    """Comparison of a single field across modes."""
    field_id: str
    field_type: str
    bounding_box: List[float]
    supporting_text: str
    results_by_mode: Dict[str, Dict[str, Any]]
    analysis: Dict[str, Any]


class ComparisonResponse(BaseModel):
    """Response for comparison endpoint."""
    success: bool
    document_id: Optional[str] = None
    page_number: int = 1
    mode_results: Dict[str, ModeResultOutput] = {}
    field_comparisons: List[FieldComparisonOutput] = []
    improvement_summary: Dict[str, Any] = {}
    total_processing_time_ms: int = 0
    error: Optional[str] = None


# ============================================================================
# Pipeline Instances (Singletons)
# ============================================================================

_pipeline_instance = None
_comparison_pipeline_instance = None


def get_pipeline():
    """Get or create the pipeline instance."""
    global _pipeline_instance
    
    if _pipeline_instance is None:
        from app.services.legal_form_pipeline import LegalFormPipeline
        
        _pipeline_instance = LegalFormPipeline(
            use_gpu=True,  # Will fall back to CPU if unavailable
            enable_glm_adjudication=True
        )
        logger.info("Initialized LegalFormPipeline singleton")
    
    return _pipeline_instance


def get_comparison_pipeline():
    """Get or create the comparison pipeline instance."""
    global _comparison_pipeline_instance
    
    if _comparison_pipeline_instance is None:
        from app.services.legal_form_pipeline import ComparisonPipeline
        
        _comparison_pipeline_instance = ComparisonPipeline(
            use_gpu=True,
        )
        logger.info("Initialized ComparisonPipeline singleton")
    
    return _comparison_pipeline_instance


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_pdf(
    file: UploadFile = File(..., description="PDF file to analyze"),
    include_statistics: bool = Query(True, description="Include statistics summary"),
    document_id: Optional[str] = Query(None, description="Optional document identifier")
) -> AnalyzeResponse:
    """
    Analyze a PDF document for empty fillable fields.
    
    This endpoint processes the PDF through the full pipeline:
    1. Textract OCR + geometry extraction
    2. Field candidate detection
    3. LayoutLM semantic classification
    4. GLM adjudication for ambiguous cases
    
    Returns detected fields with semantic labels and confidence scores.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Read file content
        pdf_bytes = await file.read()
        
        if len(pdf_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Empty file uploaded"
            )
        
        # Validate PDF magic bytes
        if not pdf_bytes.startswith(b'%PDF'):
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF format"
            )
        
        logger.info(f"Analyzing PDF: {file.filename} ({len(pdf_bytes)} bytes)")
        
        # Process through pipeline
        pipeline = get_pipeline()
        result = pipeline.process_pdf(pdf_bytes, document_id=document_id)
        
        # Build response
        pages_output = []
        for page in result.pages:
            fields_output = [
                FieldOutput(
                    field_id=f.field_id,
                    field_type=f.field_type,
                    bounding_box=f.bounding_box,
                    semantic_label=f.semantic_label,
                    confidence=f.confidence,
                    supporting_text=f.supporting_text
                )
                for f in page.fields
            ]
            pages_output.append(PageOutput(page=page.page, fields=fields_output))
        
        # Get statistics if requested
        statistics = None
        if include_statistics:
            statistics = pipeline.get_statistics(result)
        
        return AnalyzeResponse(
            success=True,
            document_id=result.document_id,
            pages=pages_output,
            statistics=statistics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return AnalyzeResponse(
            success=False,
            error=str(e)
        )


@router.post("/analyze/extended", response_model=ExtendedAnalyzeResponse)
async def analyze_pdf_extended(
    file: UploadFile = File(..., description="PDF file to analyze"),
    document_id: Optional[str] = Query(None, description="Optional document identifier")
) -> ExtendedAnalyzeResponse:
    """
    Analyze a PDF with full extended output including audit trail.
    
    This endpoint provides the complete pipeline output including:
    - Extended field metadata (detection/classification methods)
    - Processing timestamps and durations
    - GLM adjudication audit log for legal review
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        pdf_bytes = await file.read()
        
        if not pdf_bytes.startswith(b'%PDF'):
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF format"
            )
        
        pipeline = get_pipeline()
        result = pipeline.process_pdf(pdf_bytes, document_id=document_id)
        
        # Get full extended output
        extended_dict = result.to_extended_dict()
        
        return ExtendedAnalyzeResponse(
            success=True,
            document_id=result.document_id,
            pages=extended_dict['pages'],
            metadata=extended_dict['metadata'],
            audit=extended_dict['audit'],
            statistics=pipeline.get_statistics(result)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return ExtendedAnalyzeResponse(
            success=False,
            error=str(e)
        )


@router.post("/analyze-textract", response_model=AnalyzeResponse)
async def analyze_textract_output(
    request: TextractAnalyzeRequest,
    include_statistics: bool = Query(True, description="Include statistics summary")
) -> AnalyzeResponse:
    """
    Analyze pre-extracted Textract blocks.
    
    Use this endpoint when you've already called Textract separately
    and want to run the field detection + classification stages.
    
    Input should be the raw Textract blocks array.
    """
    try:
        if not request.blocks:
            raise HTTPException(
                status_code=400,
                detail="No Textract blocks provided"
            )
        
        logger.info(f"Analyzing Textract output: {len(request.blocks)} blocks")
        
        pipeline = get_pipeline()
        result = pipeline.process_textract_output(
            textract_blocks=request.blocks,
            page_number=request.page_number,
            document_id=request.document_id
        )
        
        # Build response
        pages_output = []
        for page in result.pages:
            fields_output = [
                FieldOutput(
                    field_id=f.field_id,
                    field_type=f.field_type,
                    bounding_box=f.bounding_box,
                    semantic_label=f.semantic_label,
                    confidence=f.confidence,
                    supporting_text=f.supporting_text
                )
                for f in page.fields
            ]
            pages_output.append(PageOutput(page=page.page, fields=fields_output))
        
        statistics = None
        if include_statistics:
            statistics = pipeline.get_statistics(result)
        
        return AnalyzeResponse(
            success=True,
            document_id=result.document_id,
            pages=pages_output,
            statistics=statistics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return AnalyzeResponse(
            success=False,
            error=str(e)
        )


@router.get("/ontology", response_model=OntologyResponse)
async def get_ontology() -> OntologyResponse:
    """
    Get the complete semantic label ontology.
    
    Returns all valid labels that the pipeline can assign to fields.
    Useful for understanding the classification schema and building
    downstream integrations.
    """
    from app.services.legal_form_pipeline.ontology import ONTOLOGY, SemanticLabel
    
    labels = []
    for label in SemanticLabel:
        metadata = ONTOLOGY.get_metadata(label)
        if metadata:
            labels.append(OntologyLabel(
                label=label.value,
                aliases=metadata.aliases,
                typical_field_types=metadata.typical_field_types,
                is_attestational=metadata.is_attestational
            ))
        else:
            labels.append(OntologyLabel(
                label=label.value,
                aliases=[],
                typical_field_types=["line"],
                is_attestational=False
            ))
    
    return OntologyResponse(
        total_labels=len(labels),
        labels=labels
    )


@router.post("/compare", response_model=ComparisonResponse)
async def compare_processing_modes(
    file: UploadFile = File(..., description="PDF file to analyze"),
    document_id: Optional[str] = Query(None, description="Optional document identifier")
) -> ComparisonResponse:
    """
    Compare the same form processed through different technology configurations.
    
    This endpoint processes the PDF through 4 different modes:
    1. TEXTRACT_ONLY: Just Textract OCR (baseline)
    2. TEXTRACT_HEURISTICS: Textract + rule-based classification
    3. TEXTRACT_LAYOUTLM: Textract + LayoutLMv3 classification
    4. FULL_STACK: Textract + LayoutLMv3 + GLM-4.5V adjudication
    
    Returns a side-by-side comparison showing how each technology improves results.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        pdf_bytes = await file.read()
        
        if not pdf_bytes.startswith(b'%PDF'):
            raise HTTPException(
                status_code=400,
                detail="Invalid PDF format"
            )
        
        logger.info(f"Running comparison on: {file.filename} ({len(pdf_bytes)} bytes)")
        
        # Run Textract first to get blocks
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
        
        # Extract page image first (we'll need it either way)
        page_image = None
        image_bytes = None
        try:
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, dpi=150)
            if images:
                page_image = images[0]
                # Also prepare image bytes for Textract fallback
                buffer = BytesIO()
                page_image.save(buffer, format='PNG')
                image_bytes = buffer.getvalue()
                logger.info(f"Converted PDF to image: {len(image_bytes)} bytes")
        except Exception as e:
            logger.warning(f"Could not extract page image: {e}")
        
        # Call Textract - try PDF first, fall back to image if unsupported
        textract_blocks = []
        try:
            response = client.analyze_document(
                Document={'Bytes': pdf_bytes},
                FeatureTypes=['FORMS', 'TABLES']
            )
            textract_blocks = response.get('Blocks', [])
            logger.info(f"Textract processed PDF directly: {len(textract_blocks)} blocks")
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'UnsupportedDocumentException' and image_bytes:
                # PDF format not supported, try with converted image
                logger.warning(f"PDF format not supported by Textract, using converted image")
                try:
                    response = client.analyze_document(
                        Document={'Bytes': image_bytes},
                        FeatureTypes=['FORMS', 'TABLES']
                    )
                    textract_blocks = response.get('Blocks', [])
                    logger.info(f"Textract processed image: {len(textract_blocks)} blocks")
                except Exception as img_error:
                    logger.error(f"Textract failed on image too: {img_error}")
                    raise
            else:
                # Re-raise other errors
                raise
        
        # Run comparison pipeline
        comparison_pipeline = get_comparison_pipeline()
        result = comparison_pipeline.compare(
            textract_blocks=textract_blocks,
            page_image=page_image,
            page_number=1,
            document_id=document_id or file.filename.replace('.pdf', '')
        )
        
        # Convert to response format
        mode_results_output = {}
        for mode_name, mode_result in result.mode_results.items():
            mode_dict = mode_result.to_dict()
            mode_results_output[mode_name] = ModeResultOutput(
                mode=mode_dict['mode'],
                mode_description=mode_dict['mode_description'],
                components_used=mode_dict['components_used'],
                fields=mode_dict['fields'],
                processing_time_ms=mode_dict['processing_time_ms'],
                statistics=mode_dict['statistics'],
                annotated_image=mode_dict.get('annotated_image')
            )
        
        field_comparisons_output = []
        for fc in result.field_comparisons:
            fc_dict = fc.to_dict()
            field_comparisons_output.append(FieldComparisonOutput(
                field_id=fc_dict['field_id'],
                field_type=fc_dict['field_type'],
                bounding_box=fc_dict['bounding_box'],
                supporting_text=fc_dict['supporting_text'],
                results_by_mode=fc_dict['results_by_mode'],
                analysis=fc_dict['analysis']
            ))
        
        return ComparisonResponse(
            success=True,
            document_id=result.document_id,
            page_number=result.page_number,
            mode_results=mode_results_output,
            field_comparisons=field_comparisons_output,
            improvement_summary=result.improvement_summary,
            total_processing_time_ms=result.total_processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison pipeline error: {e}", exc_info=True)
        return ComparisonResponse(
            success=False,
            error=str(e)
        )


@router.post("/compare-textract", response_model=ComparisonResponse)
async def compare_from_textract(
    request: TextractAnalyzeRequest
) -> ComparisonResponse:
    """
    Compare processing modes using pre-extracted Textract blocks.
    
    Use this when you've already called Textract separately.
    """
    try:
        if not request.blocks:
            raise HTTPException(
                status_code=400,
                detail="No Textract blocks provided"
            )
        
        logger.info(f"Running comparison on {len(request.blocks)} Textract blocks")
        
        comparison_pipeline = get_comparison_pipeline()
        result = comparison_pipeline.compare(
            textract_blocks=request.blocks,
            page_image=None,
            page_number=request.page_number,
            document_id=request.document_id or "textract_comparison"
        )
        
        # Convert to response format
        mode_results_output = {}
        for mode_name, mode_result in result.mode_results.items():
            mode_dict = mode_result.to_dict()
            mode_results_output[mode_name] = ModeResultOutput(
                mode=mode_dict['mode'],
                mode_description=mode_dict['mode_description'],
                components_used=mode_dict['components_used'],
                fields=mode_dict['fields'],
                processing_time_ms=mode_dict['processing_time_ms'],
                statistics=mode_dict['statistics'],
                annotated_image=mode_dict.get('annotated_image')
            )
        
        field_comparisons_output = []
        for fc in result.field_comparisons:
            fc_dict = fc.to_dict()
            field_comparisons_output.append(FieldComparisonOutput(
                field_id=fc_dict['field_id'],
                field_type=fc_dict['field_type'],
                bounding_box=fc_dict['bounding_box'],
                supporting_text=fc_dict['supporting_text'],
                results_by_mode=fc_dict['results_by_mode'],
                analysis=fc_dict['analysis']
            ))
        
        return ComparisonResponse(
            success=True,
            document_id=result.document_id,
            page_number=result.page_number,
            mode_results=mode_results_output,
            field_comparisons=field_comparisons_output,
            improvement_summary=result.improvement_summary,
            total_processing_time_ms=result.total_processing_time_ms
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison pipeline error: {e}", exc_info=True)
        return ComparisonResponse(
            success=False,
            error=str(e)
        )


@router.get("/comparison-modes")
async def get_comparison_modes() -> Dict[str, Any]:
    """
    Get information about available comparison modes.
    
    Returns descriptions of each technology configuration.
    """
    from app.services.legal_form_pipeline import ComparisonPipeline, ComparisonMode
    
    return {
        "modes": [
            {
                "mode": ComparisonMode.TEXTRACT_ONLY.value,
                "name": "Textract Only",
                "description": ComparisonPipeline.MODE_DESCRIPTIONS[ComparisonMode.TEXTRACT_ONLY],
                "components": ComparisonPipeline.MODE_COMPONENTS[ComparisonMode.TEXTRACT_ONLY]
            },
            {
                "mode": ComparisonMode.TEXTRACT_HEURISTICS.value,
                "name": "Textract + Heuristics",
                "description": ComparisonPipeline.MODE_DESCRIPTIONS[ComparisonMode.TEXTRACT_HEURISTICS],
                "components": ComparisonPipeline.MODE_COMPONENTS[ComparisonMode.TEXTRACT_HEURISTICS]
            },
            {
                "mode": ComparisonMode.TEXTRACT_LAYOUTLM.value,
                "name": "Textract + LayoutLM",
                "description": ComparisonPipeline.MODE_DESCRIPTIONS[ComparisonMode.TEXTRACT_LAYOUTLM],
                "components": ComparisonPipeline.MODE_COMPONENTS[ComparisonMode.TEXTRACT_LAYOUTLM]
            },
            {
                "mode": ComparisonMode.FULL_STACK.value,
                "name": "Full Stack",
                "description": ComparisonPipeline.MODE_DESCRIPTIONS[ComparisonMode.FULL_STACK],
                "components": ComparisonPipeline.MODE_COMPONENTS[ComparisonMode.FULL_STACK]
            }
        ]
    }


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check for the legal form pipeline.
    
    Returns status of all pipeline components.
    """
    try:
        pipeline = get_pipeline()
        comparison_pipeline = get_comparison_pipeline()
        
        return {
            "status": "healthy",
            "components": {
                "field_detector": "ready",
                "layoutlm_classifier": {
                    "status": "ready",
                    "mode": pipeline.classifier.mode
                },
                "glm_adjudicator": {
                    "status": "ready" if pipeline.adjudicator and pipeline.adjudicator.is_available else "fallback",
                    "api_available": pipeline.adjudicator.is_available if pipeline.adjudicator else False
                },
                "comparison_pipeline": "ready"
            },
            "ontology_labels": pipeline.ontology.num_labels,
            "comparison_modes_available": 4
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

