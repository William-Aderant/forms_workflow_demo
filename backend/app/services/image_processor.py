"""
Image processing service for creating annotated images with OCR bounding boxes.
"""
import logging
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from app.config import Config
from app.utils.pdf_handler import PDFHandler

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Service for processing images and drawing OCR bounding boxes."""
    
    def __init__(self):
        """Initialize image processor."""
        self.pdf_handler = PDFHandler()
        self.output_dir = Path(Config.IMAGES_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Image processor initialized. Output directory: {self.output_dir}")
    
    def create_annotated_image(
        self,
        pdf_bytes: bytes,
        bounding_boxes: List[Dict[str, Any]],
        form_id: str,
        form_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Create an annotated image from PDF with OCR bounding boxes drawn.
        
        Args:
            pdf_bytes: PDF file as bytes
            bounding_boxes: List of bounding box dictionaries from Textract
            form_id: Unique form identifier
            form_name: Form name
        
        Returns:
            Dictionary with image path and metadata, or None if failed
        """
        try:
            # Convert PDF first page to image
            # #region agent log
            import json
            import os
            log_path = '/Users/william.holden/Documents/forms_workflow_demo/.cursor/debug.log'
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a') as f:
                f.write(json.dumps({"location":"image_processor.py:48","message":"Attempting to convert PDF to image for annotation","data":{"form_id":form_id,"pdf_size":len(pdf_bytes)},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"IMAGE_PROCESSOR_PDF_CONVERSION"})+"\n")
            # #endregion
            
            images = self.pdf_handler.pdf_to_images(pdf_bytes, first_page_only=True)
            if not images:
                logger.warning(f"Failed to convert PDF to image for form {form_id} (poppler may not be installed). Skipping image annotation but OCR processing continues.")
                # #region agent log
                with open(log_path, 'a') as f:
                    f.write(json.dumps({"location":"image_processor.py:52","message":"PDF to image conversion failed","data":{"form_id":form_id,"reason":"poppler_not_available"},"timestamp":int(__import__('time').time()*1000),"sessionId":"debug-session","runId":"run1","hypothesisId":"IMAGE_PROCESSOR_PDF_CONVERSION"})+"\n")
                # #endregion
                return None
            
            image = images[0]
            
            # Get image dimensions
            img_width, img_height = image.size
            
            # Create a copy for annotation
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            # Draw bounding boxes
            drawn_boxes = []
            for bbox_info in bounding_boxes:
                bbox = bbox_info.get('bounding_box', {})
                if not bbox:
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                left = bbox.get('left', 0) * img_width
                top = bbox.get('top', 0) * img_height
                width = bbox.get('width', 0) * img_width
                height = bbox.get('height', 0) * img_height
                
                # Calculate rectangle coordinates
                x1 = left
                y1 = top
                x2 = left + width
                y2 = top + height
                
                # Draw rectangle (red outline)
                draw.rectangle(
                    [(x1, y1), (x2, y2)],
                    outline='red',
                    width=2
                )
                
                # Draw text label if available
                text = bbox_info.get('text', '')
                if text and font:
                    # Draw text background
                    text_bbox = draw.textbbox((x1, y1 - 15), text, font=font)
                    if text_bbox:
                        draw.rectangle(
                            text_bbox,
                            fill='red',
                            outline='red'
                        )
                        # Draw text
                        draw.text(
                            (x1, y1 - 15),
                            text[:30],  # Truncate long text
                            fill='white',
                            font=font
                        )
                
                # Store box information
                drawn_boxes.append({
                    'block_type': bbox_info.get('block_type', ''),
                    'text': text,
                    'confidence': bbox_info.get('confidence', 0),
                    'bounding_box': {
                        'left': bbox.get('left', 0),
                        'top': bbox.get('top', 0),
                        'width': bbox.get('width', 0),
                        'height': bbox.get('height', 0)
                    },
                    'pixel_coordinates': {
                        'x1': int(x1),
                        'y1': int(y1),
                        'x2': int(x2),
                        'y2': int(y2)
                    }
                })
            
            # Save annotated image
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_form_name = "".join(c for c in form_name if c.isalnum() or c in (' ', '-', '_')).strip()[:50].replace(' ', '_')
            image_filename = f"{form_id}_{safe_form_name}_{timestamp}.png"
            image_path = self.output_dir / image_filename
            
            annotated_image.save(image_path, 'PNG')
            logger.info(f"Saved annotated image: {image_path}")
            
            # Create metadata
            metadata = {
                'image_path': image_filename,  # Store just filename for API serving
                'form_id': form_id,
                'form_name': form_name,
                'image_dimensions': {
                    'width': img_width,
                    'height': img_height
                },
                'boxes': drawn_boxes,
                'total_boxes': len(drawn_boxes),
                'created_at': datetime.now().isoformat()
            }
            
            # Save JSON metadata
            json_filename = f"{form_id}_{safe_form_name}_{timestamp}.json"
            json_path = self.output_dir / json_filename
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata: {json_path}")
            
            return {
                'image_path': image_filename,  # Return filename for API serving
                'json_path': str(json_path),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error creating annotated image for form {form_id}: {e}")
            return None
    
    def process_sample_forms(
        self,
        forms: List[Dict[str, Any]],
        max_samples: int = None
    ) -> List[Dict[str, Any]]:
        """
        Process sample forms and create annotated images.
        
        Args:
            forms: List of form dictionaries with pdf_bytes, bounding_boxes, form_id, form_name
            max_samples: Maximum number of forms to process (defaults to Config.SAMPLE_IMAGES_COUNT)
        
        Returns:
            List of processed image metadata dictionaries
        """
        max_samples = max_samples or Config.SAMPLE_IMAGES_COUNT
        processed = []
        
        for i, form in enumerate(forms[:max_samples]):
            if i >= max_samples:
                break
            
            result = self.create_annotated_image(
                pdf_bytes=form.get('pdf_bytes'),
                bounding_boxes=form.get('bounding_boxes', []),
                form_id=form.get('form_id'),
                form_name=form.get('form_name', 'Unknown')
            )
            
            if result:
                processed.append(result)
        
        logger.info(f"Processed {len(processed)} sample images")
        return processed

