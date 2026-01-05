"""
AWS Textract service for OCR processing of PDF forms.
"""
import logging
import boto3
from typing import Dict, Any, Optional, List
from botocore.exceptions import ClientError

from app.config import Config
from app.utils.rate_limiter import RateLimiter
from app.utils.pdf_handler import PDFHandler

logger = logging.getLogger(__name__)


class TextractService:
    """Service for processing documents with AWS Textract."""
    
    def __init__(self, rate_limiter: Optional[RateLimiter] = None):
        """
        Initialize Textract service.
        
        Args:
            rate_limiter: Optional rate limiter instance
        """
        self.rate_limiter = rate_limiter
        self.service_name = 'textract'
        self.pdf_handler = PDFHandler()
        
        # Initialize Textract client
        try:
            config = Config.get_boto3_config()
            if 'profile_name' in config:
                session = boto3.Session(profile_name=config['profile_name'])
                self.client = session.client('textract', region_name=config['region_name'])
            else:
                self.client = boto3.client('textract', **config)
            
            logger.info("Initialized AWS Textract service")
        except Exception as e:
            logger.error(f"Failed to initialize Textract client: {e}")
            raise
    
    def process_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Process a PDF document with AWS Textract.
        
        Args:
            pdf_bytes: PDF file as bytes
        
        Returns:
            Dictionary with extracted text, markdown, bounding boxes, and metadata
        """
        # Check rate limits
        if self.rate_limiter:
            can_call, reason = self.rate_limiter.can_make_call(self.service_name)
            if not can_call:
                logger.warning(f"Rate limit exceeded: {reason}")
                return {
                    'success': False,
                    'error': f"Rate limit exceeded: {reason}",
                    'text': '',
                    'markdown': '',
                    'blocks': [],
                    'bounding_boxes': []
                }
            
            # Wait if needed
            self.rate_limiter.wait_if_needed(self.service_name, min_delay_seconds=1.0)
        
        try:
            logger.info(f"Processing PDF with Textract ({len(pdf_bytes)} bytes)")
            
            # #region agent log
            import json
            import os
            log_path = '/Users/william.holden/Documents/forms_workflow_demo/.cursor/debug.log'
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            # Check if bytes are actually a PDF (check magic bytes)
            is_pdf = pdf_bytes[:4] == b'%PDF'
            with open(log_path, 'a') as f:
                f.write(json.dumps({"location":"textract_service.py:72","message":"About to call Textract","data":{"pdf_size":len(pdf_bytes),"is_pdf":is_pdf,"first_bytes":pdf_bytes[:20].hex() if len(pdf_bytes) >= 20 else "too_short"},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"PDF_VALIDATION"})+"\n")
            # #endregion
            
            if not is_pdf:
                logger.error("Bytes do not appear to be a valid PDF (missing PDF magic bytes)")
                return {
                    'success': False,
                    'error': 'Invalid PDF format: missing PDF magic bytes',
                    'text': '',
                    'markdown': '',
                    'blocks': [],
                    'bounding_boxes': []
                }
            
            # Try to use Textract's native PDF support first (no poppler dependency)
            # If PDF is too large (>5MB), we'll need to convert to image
            # #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"location":"textract_service.py:95","message":"Attempting Textract with PDF directly","data":{"pdf_size":len(pdf_bytes),"pdf_size_mb":round(len(pdf_bytes)/(1024*1024),2)},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"TEXTRACT_PDF_DIRECT"})+"\n")
            # #endregion
            
            # Check PDF size - detect_document_text has a 5MB limit for synchronous calls
            pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
            use_pdf_directly = pdf_size_mb <= 5.0
            
            if use_pdf_directly:
                # Try PDF directly first (no poppler needed)
                try:
                    # #region agent log
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"location":"textract_service.py:105","message":"Calling detect_document_text with PDF","data":{"pdf_size":len(pdf_bytes)},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"TEXTRACT_PDF_DIRECT"})+"\n")
                    # #endregion
                    
                    response = self.client.detect_document_text(
                        Document={'Bytes': pdf_bytes}
                    )
                    
                    # #region agent log
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"location":"textract_service.py:112","message":"Textract PDF call successful","data":{"blocks_count":len(response.get('Blocks', []))},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"TEXTRACT_PDF_DIRECT"})+"\n")
                    # #endregion
                except Exception as pdf_error:
                    # PDF format not supported, fall back to image conversion
                    logger.warning(f"Textract failed with PDF directly: {pdf_error}. Falling back to image conversion.")
                    # #region agent log
                    with open(log_path, 'a') as f:
                        f.write(json.dumps({"location":"textract_service.py:118","message":"PDF direct failed, converting to image","data":{"error":str(pdf_error)},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"FALLBACK_TO_IMAGE"})+"\n")
                    # #endregion
                    use_pdf_directly = False
            
            if not use_pdf_directly:
                # Convert PDF to image (requires poppler)
                # #region agent log
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"location":"textract_service.py:125","message":"Converting PDF to image","data":{"pdf_size":len(pdf_bytes)},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"PDF_TO_IMAGE"})+"\n")
                # #endregion
                
                images = self.pdf_handler.pdf_to_images(pdf_bytes, first_page_only=True)
                if not images:
                    logger.error("Failed to convert PDF to image (poppler may not be installed)")
                    return {
                        'success': False,
                        'error': 'Failed to convert PDF to image. Poppler may not be installed. Install with: brew install poppler (macOS) or apt-get install poppler-utils (Linux)',
                        'text': '',
                        'markdown': '',
                        'blocks': [],
                        'bounding_boxes': []
                    }
                
                # Convert first page image to bytes
                image_bytes = self.pdf_handler.image_to_bytes(images[0], format='PNG')
                logger.info(f"Converted PDF to image: {len(image_bytes)} bytes")
                
                # #region agent log
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"location":"textract_service.py:140","message":"Calling detect_document_text on image","data":{"image_size":len(image_bytes)},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"TEXTRACT_CALL"})+"\n")
                # #endregion
                
                # Call Textract detect_document_text on the image
                response = self.client.detect_document_text(
                    Document={'Bytes': image_bytes}
                )
                
                # #region agent log
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"location":"textract_service.py:148","message":"Textract image call successful","data":{"blocks_count":len(response.get('Blocks', []))},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"TEXTRACT_CALL"})+"\n")
                # #endregion
            
            # #region agent log
            with open(log_path, 'a') as f:
                f.write(json.dumps({"location":"textract_service.py:118","message":"Textract call successful","data":{"blocks_count":len(response.get('Blocks', []))},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"TEXTRACT_CALL"})+"\n")
            # #endregion
            
            # Record the call
            if self.rate_limiter:
                self.rate_limiter.record_call(self.service_name)
            
            # Extract text blocks and bounding boxes
            text_blocks = []
            bounding_boxes = []
            
            for block in response.get('Blocks', []):
                block_type = block.get('BlockType', '')
                
                if block_type == 'LINE':
                    text = block.get('Text', '')
                    if text:
                        text_blocks.append(text)
                
                # Collect bounding box information for all blocks
                if 'Geometry' in block and 'BoundingBox' in block['Geometry']:
                    bbox = block['Geometry']['BoundingBox']
                    bounding_boxes.append({
                        'block_type': block_type,
                        'text': block.get('Text', ''),
                        'confidence': block.get('Confidence', 0),
                        'bounding_box': {
                            'left': bbox.get('Left', 0),
                            'top': bbox.get('Top', 0),
                            'width': bbox.get('Width', 0),
                            'height': bbox.get('Height', 0)
                        }
                    })
            
            # Convert to markdown (simple format - one line per text block)
            markdown_text = '\n'.join(text_blocks)
            
            # Calculate average confidence
            confidences = [
                block.get('Confidence', 0) 
                for block in response.get('Blocks', [])
                if 'Confidence' in block
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Get document metadata
            doc_metadata = response.get('DocumentMetadata', {})
            pages = doc_metadata.get('Pages', 1)
            
            result = {
                'success': True,
                'text': '\n'.join(text_blocks),
                'markdown': markdown_text,
                'blocks': text_blocks,
                'bounding_boxes': bounding_boxes,
                'metadata': {
                    'blocks_count': len(response.get('Blocks', [])),
                    'confidence': avg_confidence,
                    'pages': pages
                }
            }
            
            logger.info(
                f"Textract processing complete: {len(text_blocks)} text blocks, "
                f"{len(bounding_boxes)} bounding boxes, confidence: {avg_confidence:.2f}%"
            )
            
            return result
            
        except ClientError as e:
            error_msg = str(e)
            logger.error(f"Textract error: {e}")
            
            # Still record the call attempt
            if self.rate_limiter:
                self.rate_limiter.record_call(self.service_name)
            
            # Provide helpful error messages
            if "ExpiredTokenException" in error_msg or "expired" in error_msg.lower():
                error_msg = (
                    f"AWS credentials have expired. {error_msg}\n"
                    "Please refresh your AWS credentials."
                )
            elif "InvalidClientTokenId" in error_msg:
                error_msg = (
                    f"AWS credentials are invalid. {error_msg}\n"
                    "Please check your AWS credentials."
                )
            
            return {
                'success': False,
                'error': error_msg,
                'text': '',
                'markdown': '',
                'blocks': [],
                'bounding_boxes': []
            }
        except Exception as e:
            logger.error(f"Unexpected error processing PDF with Textract: {e}")
            if self.rate_limiter:
                self.rate_limiter.record_call(self.service_name)
            
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'markdown': '',
                'blocks': [],
                'bounding_boxes': []
            }
    
    def get_form_fields(self, bounding_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract form fields from bounding boxes.
        Filters for KEY_VALUE_SET blocks which represent form fields.
        
        Args:
            bounding_boxes: List of bounding box dictionaries
        
        Returns:
            List of form field dictionaries
        """
        form_fields = []
        
        for bbox in bounding_boxes:
            if bbox['block_type'] in ['KEY_VALUE_SET', 'CELL']:
                form_fields.append({
                    'text': bbox['text'],
                    'confidence': bbox['confidence'],
                    'bounding_box': bbox['bounding_box']
                })
        
        return form_fields

