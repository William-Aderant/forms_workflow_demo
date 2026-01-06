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
from app.services.field_classifier import FieldClassifier

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
        self.field_classifier = FieldClassifier()
        
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
    
    def process_pdf_with_forms(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """
        Process a PDF document with AWS Textract using FORMS feature.
        This method specifically detects form fields (key-value pairs).
        
        Args:
            pdf_bytes: PDF file as bytes
        
        Returns:
            Dictionary with extracted text, markdown, form fields, and metadata
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
                    'bounding_boxes': [],
                    'form_fields': []
                }
            
            # Wait if needed
            self.rate_limiter.wait_if_needed(self.service_name, min_delay_seconds=1.0)
        
        try:
            logger.info(f"Processing PDF with Textract FORMS feature ({len(pdf_bytes)} bytes)")
            
            # Check if bytes are actually a PDF
            is_pdf = pdf_bytes[:4] == b'%PDF'
            if not is_pdf:
                logger.error("Bytes do not appear to be a valid PDF (missing PDF magic bytes)")
                return {
                    'success': False,
                    'error': 'Invalid PDF format: missing PDF magic bytes',
                    'text': '',
                    'markdown': '',
                    'blocks': [],
                    'bounding_boxes': [],
                    'form_fields': []
                }
            
            # Check PDF size - analyze_document has a 5MB limit for synchronous calls
            pdf_size_mb = len(pdf_bytes) / (1024 * 1024)
            use_pdf_directly = pdf_size_mb <= 5.0
            
            if use_pdf_directly:
                # Try PDF directly first
                try:
                    response = self.client.analyze_document(
                        Document={'Bytes': pdf_bytes},
                        FeatureTypes=['FORMS']
                    )
                    logger.info("Textract FORMS analysis successful with PDF directly")
                except Exception as pdf_error:
                    # PDF format not supported, fall back to image conversion
                    logger.warning(f"Textract failed with PDF directly: {pdf_error}. Falling back to image conversion.")
                    use_pdf_directly = False
            
            if not use_pdf_directly:
                # Convert PDF to image (requires poppler)
                images = self.pdf_handler.pdf_to_images(pdf_bytes, first_page_only=True)
                if not images:
                    logger.error("Failed to convert PDF to image (poppler may not be installed)")
                    return {
                        'success': False,
                        'error': 'Failed to convert PDF to image. Poppler may not be installed. Install with: brew install poppler (macOS) or apt-get install poppler-utils (Linux)',
                        'text': '',
                        'markdown': '',
                        'blocks': [],
                        'bounding_boxes': [],
                        'form_fields': []
                    }
                
                # Convert first page image to bytes
                image_bytes = self.pdf_handler.image_to_bytes(images[0], format='PNG')
                logger.info(f"Converted PDF to image: {len(image_bytes)} bytes")
                
                # Call Textract analyze_document with FORMS feature on the image
                response = self.client.analyze_document(
                    Document={'Bytes': image_bytes},
                    FeatureTypes=['FORMS']
                )
                logger.info("Textract FORMS analysis successful on image")
            
            # Record the call
            if self.rate_limiter:
                self.rate_limiter.record_call(self.service_name)
            
            # Extract blocks and build block map for relationship resolution
            blocks = response.get('Blocks', [])
            block_map = {block['Id']: block for block in blocks}
            
            # Extract text blocks and bounding boxes (for backward compatibility)
            text_blocks = []
            bounding_boxes = []
            
            for block in blocks:
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
            
            # Extract form fields (KEY_VALUE_SET blocks)
            form_fields = self._extract_form_fields(blocks, block_map)
            
            # Convert to markdown
            markdown_text = '\n'.join(text_blocks)
            
            # Calculate average confidence
            confidences = [
                block.get('Confidence', 0) 
                for block in blocks
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
                'form_fields': form_fields,
                'metadata': {
                    'blocks_count': len(blocks),
                    'confidence': avg_confidence,
                    'pages': pages,
                    'form_fields_count': len(form_fields)
                }
            }
            
            logger.info(
                f"Textract FORMS processing complete: {len(text_blocks)} text blocks, "
                f"{len(form_fields)} form fields, confidence: {avg_confidence:.2f}%"
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
                'bounding_boxes': [],
                'form_fields': []
            }
        except Exception as e:
            logger.error(f"Unexpected error processing PDF with Textract FORMS: {e}")
            if self.rate_limiter:
                self.rate_limiter.record_call(self.service_name)
            
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'markdown': '',
                'blocks': [],
                'bounding_boxes': [],
                'form_fields': []
            }
    
    def _extract_form_fields(self, blocks: List[Dict[str, Any]], block_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract form fields from KEY_VALUE_SET blocks.
        Identifies VALUE blocks (input areas) and finds the closest text label using spatial proximity.
        
        Args:
            blocks: List of all blocks from Textract response
            block_map: Dictionary mapping block IDs to blocks
        
        Returns:
            List of form field dictionaries with VALUE bounding boxes and closest text labels
        """
        import math
        
        form_fields = []
        
        # Collect all LINE blocks with their positions and text for spatial label matching
        # LINE blocks are better than WORD blocks because they preserve the full label text
        line_blocks = []
        for block in blocks:
            if block.get('BlockType') == 'LINE':
                geometry = block.get('Geometry', {})
                bbox = geometry.get('BoundingBox', {})
                text = block.get('Text', '').strip()
                if bbox and text:
                    line_blocks.append({
                        'text': text,
                        'left': bbox.get('Left', 0),
                        'top': bbox.get('Top', 0),
                        'right': bbox.get('Left', 0) + bbox.get('Width', 0),
                        'bottom': bbox.get('Top', 0) + bbox.get('Height', 0),
                        'center_x': bbox.get('Left', 0) + bbox.get('Width', 0) / 2,
                        'center_y': bbox.get('Top', 0) + bbox.get('Height', 0) / 2,
                    })
        
        # Find all KEY_VALUE_SET blocks
        key_value_blocks = [block for block in blocks if block.get('BlockType') == 'KEY_VALUE_SET']
        
        for kv_block in key_value_blocks:
            entity_types = kv_block.get('EntityTypes', [])
            
            # We're interested in VALUE blocks (input areas)
            if 'VALUE' in entity_types:
                geometry = kv_block.get('Geometry', {})
                bbox = geometry.get('BoundingBox', {})
                
                if not bbox:
                    continue
                
                # Get VALUE block coordinates
                value_left = bbox.get('Left', 0)
                value_top = bbox.get('Top', 0)
                value_right = value_left + bbox.get('Width', 0)
                value_bottom = value_top + bbox.get('Height', 0)
                value_center_y = value_top + bbox.get('Height', 0) / 2
                
                # Extract any existing value text from child WORD blocks
                value_text = ''
                relationships = kv_block.get('Relationships', [])
                for relationship in relationships:
                    if relationship.get('Type') == 'CHILD':
                        value_words = []
                        for child_id in relationship.get('Ids', []):
                            child_block = block_map.get(child_id)
                            if child_block and child_block.get('BlockType') == 'WORD':
                                value_words.append(child_block.get('Text', ''))
                        value_text = ' '.join(value_words)
                
                # SPATIAL LABEL DETECTION: Find the closest text that could be a label
                # Labels are typically to the LEFT or ABOVE the input field
                best_label = None
                best_score = float('inf')
                
                for line in line_blocks:
                    # Skip if the line overlaps significantly with the VALUE box
                    # (it might be the value itself, not a label)
                    horizontal_overlap = max(0, min(line['right'], value_right) - max(line['left'], value_left))
                    vertical_overlap = max(0, min(line['bottom'], value_bottom) - max(line['top'], value_top))
                    overlap_area = horizontal_overlap * vertical_overlap
                    value_area = bbox.get('Width', 0.01) * bbox.get('Height', 0.01)
                    
                    if overlap_area > value_area * 0.3:  # Skip if >30% overlap
                        continue
                    
                    # Calculate position relative to VALUE box
                    is_left_of = line['right'] <= value_left + 0.02  # To the left (with small tolerance)
                    is_above = line['bottom'] <= value_top + 0.02  # Above (with small tolerance)
                    is_same_row = abs(line['center_y'] - value_center_y) < 0.03  # Same horizontal row
                    
                    # Skip text that is to the RIGHT of or BELOW the field (not labels)
                    if line['left'] > value_right + 0.05:
                        continue
                    if line['top'] > value_bottom + 0.02:
                        continue
                    
                    # Calculate distance score (lower is better)
                    # Prefer text that is: 1) on the same row to the left, 2) directly above
                    if is_left_of and is_same_row:
                        # Best case: label is directly to the left on the same row
                        horizontal_dist = value_left - line['right']
                        score = horizontal_dist * 0.5  # Low multiplier = high priority
                    elif is_above and abs(line['left'] - value_left) < 0.15:
                        # Good case: label is above and roughly aligned
                        vertical_dist = value_top - line['bottom']
                        horizontal_offset = abs(line['left'] - value_left)
                        score = vertical_dist + horizontal_offset * 2
                    elif is_left_of:
                        # Label is to the left but not on same row
                        horizontal_dist = value_left - line['right']
                        vertical_dist = abs(line['center_y'] - value_center_y)
                        score = horizontal_dist + vertical_dist * 3
                    elif is_above:
                        # Label is above but not aligned
                        vertical_dist = value_top - line['bottom']
                        horizontal_offset = abs(line['left'] - value_left)
                        score = vertical_dist * 2 + horizontal_offset * 3
                    else:
                        # Not a good candidate for a label
                        continue
                    
                    # Update best match if this is closer
                    if score < best_score and score < 0.25:  # Max distance threshold
                        best_score = score
                        best_label = line['text']
                
                # Use the spatially-detected label
                key_text = best_label or ''
                
                # Classify the field type
                field_type, field_confidence = self.field_classifier.classify_field(key_text)
                
                # Log if we couldn't find key_text for debugging
                if not key_text:
                    logger.debug(f"Could not find nearby label for VALUE block at ({value_left:.3f}, {value_top:.3f})")
                
                form_fields.append({
                    'label_text': key_text,
                    'value_text': value_text,
                    'field_type': field_type,
                    'bounding_box': {
                        'left': bbox.get('Left', 0),
                        'top': bbox.get('Top', 0),
                        'width': bbox.get('Width', 0),
                        'height': bbox.get('Height', 0)
                    },
                    'confidence': kv_block.get('Confidence', 0),
                    'field_confidence': field_confidence,
                    'block_id': kv_block.get('Id', '')
                })
        
        logger.info(f"Extracted {len(form_fields)} form fields from KEY_VALUE_SET blocks")
        return form_fields
    
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

