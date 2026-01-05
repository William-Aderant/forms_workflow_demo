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


class ImageMetadata(BaseModel):
    """Image metadata with bounding box information."""
    image_path: str
    form_id: str
    form_name: str
    boxes: List[Dict[str, Any]] = Field(..., description="List of bounding boxes with coordinates and text")
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




