"""
Pydantic models for API request/response schemas.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class RateLimitStatus(BaseModel):
    """Rate limit status response model."""
    total_calls: int
    max_calls: int
    remaining_calls: int
    calls_by_service: Dict[str, int]


class FormInfo(BaseModel):
    """Form information model."""
    id: str
    name: str
    url: str
    status: str = Field(..., description="Status: pending, processing, completed, error")
    ocr_confidence: Optional[float] = None
    text_length: Optional[int] = None
    created_at: datetime
    error_message: Optional[str] = None


class FormDetail(BaseModel):
    """Detailed form information with OCR results."""
    id: str
    name: str
    url: str
    status: str
    markdown_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    text_length: Optional[int] = None
    blocks_count: Optional[int] = None
    created_at: datetime
    error_message: Optional[str] = None


class FormField(BaseModel):
    """Form field model with field type and bounding box information."""
    field_type: str = Field(..., description="Detected field type (e.g., 'name', 'address', 'attorney')")
    label_text: str = Field(..., description="Label text from the form (e.g., 'Name:', 'Address')")
    value_text: Optional[str] = Field(None, description="Value text if form is filled (empty for blank forms)")
    bounding_box: Dict[str, float] = Field(..., description="Bounding box coordinates (left, top, width, height, normalized 0-1)")
    confidence: float = Field(..., description="OCR confidence score (0-100)")
    field_confidence: float = Field(..., description="Field type classification confidence (0-1)")


class ImageMetadata(BaseModel):
    """Image metadata with bounding box information."""
    image_path: str
    form_id: str
    form_name: str
    boxes: List[Dict[str, Any]] = Field(default_factory=list, description="List of bounding boxes with coordinates and text (legacy)")
    form_fields: List[FormField] = Field(default_factory=list, description="List of detected form fields with field types")
    created_at: datetime


class ScrapeRequest(BaseModel):
    """Request model for starting scrape operation."""
    url: str = Field(default="https://courts.ca.gov/rules-forms/court-forms", description="URL to scrape")


class ScrapeResponse(BaseModel):
    """Response model for scrape operation."""
    message: str
    job_id: Optional[str] = None
    estimated_forms: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime




