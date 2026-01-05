"""
PDF handling utilities for in-memory processing.
Converts PDFs to images without saving to disk.
"""
import logging
from typing import Optional, List
from io import BytesIO
import requests
from pdf2image import convert_from_bytes
from PIL import Image

logger = logging.getLogger(__name__)


class PDFHandler:
    """Handler for PDF processing in memory."""
    
    @staticmethod
    def download_pdf(url: str) -> Optional[bytes]:
        """
        Download PDF from URL and return as bytes.
        
        Args:
            url: URL of the PDF to download
        
        Returns:
            PDF file as bytes, or None if download fails
        """
        try:
            logger.info(f"Downloading PDF from {url}")
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check if content is actually a PDF
            content_type = response.headers.get('Content-Type', '').lower()
            if 'pdf' not in content_type:
                logger.warning(f"URL {url} does not appear to be a PDF (Content-Type: {content_type})")
                # Still try to process it, might be a PDF with wrong headers
            
            pdf_bytes = response.content
            logger.info(f"Downloaded PDF: {len(pdf_bytes)} bytes")
            return pdf_bytes
            
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {e}")
            return None
    
    @staticmethod
    def pdf_to_images(pdf_bytes: bytes, first_page_only: bool = True) -> List[Image.Image]:
        """
        Convert PDF bytes to PIL Image objects.
        
        Args:
            pdf_bytes: PDF file as bytes
            first_page_only: If True, only convert first page
        
        Returns:
            List of PIL Image objects
        """
        try:
            if first_page_only:
                images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
            else:
                images = convert_from_bytes(pdf_bytes)
            
            logger.info(f"Converted PDF to {len(images)} image(s)")
            return images
            
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []
    
    @staticmethod
    def image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
        """
        Convert PIL Image to bytes.
        
        Args:
            image: PIL Image object
            format: Image format (PNG, JPEG, etc.)
        
        Returns:
            Image as bytes
        """
        buffer = BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()




