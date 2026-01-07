"""
Legal Form Understanding Pipeline
================================

A production-grade, legally-auditable pipeline for detecting and classifying
empty fillable fields in court forms.

Pipeline Stages:
1. INPUT: PDF normalization and page splitting
2. OCR + GEOMETRY: Textract extraction (WORD, LINE, SELECTION_ELEMENT)
3. FIELD CANDIDATE DETECTION: Empty lines, checkboxes, table cells
4. SEMANTIC LABELING: LayoutLMv3 classification
5. AMBIGUITY HANDLING: GLM-4.5V adjudication
6. OUTPUT: Deterministic JSON schema

Design Principles:
- Pixel-accurate bounding boxes (geometry always from Textract)
- Deterministic outputs (same input = same output)
- Minimal hallucination (strict ontology enforcement)
- Clear confidence scoring
- Legal defensibility (full audit trail)

Comparison Pipeline:
- Compare results across different technology configurations
- Modes: TEXTRACT_ONLY, TEXTRACT_HEURISTICS, TEXTRACT_LAYOUTLM, FULL_STACK
- Shows incremental value of each component
"""

from .ontology import LegalFieldOntology, SemanticLabel
from .field_candidates import FieldCandidateDetector, FieldCandidate, FieldType
from .layoutlm_classifier import LayoutLMClassifier
from .glm_adjudicator import GLMAdjudicator
from .pipeline import LegalFormPipeline, PipelineOutput, FieldResult
from .comparison_pipeline import (
    ComparisonPipeline,
    ComparisonOutput,
    ComparisonMode,
    ModeResult,
    FieldComparison
)

__all__ = [
    'LegalFormPipeline',
    'PipelineOutput',
    'FieldResult',
    'LegalFieldOntology',
    'SemanticLabel',
    'FieldCandidateDetector',
    'FieldCandidate',
    'FieldType',
    'LayoutLMClassifier',
    'GLMAdjudicator',
    # Comparison pipeline
    'ComparisonPipeline',
    'ComparisonOutput',
    'ComparisonMode',
    'ModeResult',
    'FieldComparison',
]

