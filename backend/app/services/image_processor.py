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
from app.services.field_classifier import FieldClassifier

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Service for processing images and drawing OCR bounding boxes."""
    
    def __init__(self):
        """Initialize image processor."""
        self.pdf_handler = PDFHandler()
        self.output_dir = Path(Config.IMAGES_OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.field_classifier = FieldClassifier()
        logger.info(f"Image processor initialized. Output directory: {self.output_dir}")
    
    def _get_field_color(self, field_type: str) -> tuple:
        """
        Get color for a field type.
        
        Args:
            field_type: Field type identifier
        
        Returns:
            Tuple of (outline_color, fill_color) as RGB tuples
        """
        color_map = {
            'name': ('#E63946', '#FFE5E8'),  # Red with light pink fill
            'address': ('#2A9D8F', '#E0F7F4'),  # Teal with light teal fill
            'attorney': ('#1E88E5', '#E3F2FD'),  # Blue with light blue fill
            'date': ('#F77F00', '#FFF4E6'),  # Orange with light orange fill
            'phone': ('#06A77D', '#E0F5F0'),  # Green with light green fill
            'email': ('#F4D03F', '#FEF9E7'),  # Yellow with light yellow fill
            'signature': ('#9B59B6', '#F4ECF7'),  # Purple with light purple fill
            'case_number': ('#3498DB', '#EBF5FB'),  # Sky blue with light blue fill
            'court': ('#E67E22', '#FEF5E7'),  # Orange with light orange fill
            'plaintiff': ('#E74C3C', '#FADBD8'),  # Red with light red fill
            'defendant': ('#3498DB', '#D6EAF8'),  # Blue with light blue fill
            'description': ('#8E44AD', '#F4ECF7'),  # Purple with light purple fill
            'location': ('#16A085', '#D5F4E6'),  # Dark teal with light teal fill
            'amount': ('#D35400', '#FDEBD0'),  # Dark orange with light orange fill
            'other': ('#5D6D7E', '#E8EAED'),  # Neutral slate with light gray fill
            'unknown': ('#7F8C8D', '#F4F6F7'),  # Gray with light gray fill
        }
        default = ('#5D6D7E', '#E8EAED')  # Neutral slate for unmatched field types
        outline, fill = color_map.get(field_type, default)
        
        # Convert hex to RGB tuples
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        return (hex_to_rgb(outline), hex_to_rgb(fill))
    
    def create_annotated_image(
        self,
        pdf_bytes: bytes,
        bounding_boxes: List[Dict[str, Any]] = None,
        form_fields: List[Dict[str, Any]] = None,
        form_id: str = None,
        form_name: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create an annotated image from PDF with form field bounding boxes drawn.
        
        Args:
            pdf_bytes: PDF file as bytes
            bounding_boxes: List of bounding box dictionaries from Textract (legacy, optional)
            form_fields: List of form field dictionaries with field types (preferred)
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
            # Use larger, more readable fonts
            try:
                # Try system fonts first (macOS)
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
                label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
                try:
                    bold_font = ImageFont.truetype("/System/Library/Fonts/Helvetica-Bold.ttc", 13)
                except:
                    bold_font = label_font  # Fallback to regular font if bold not available
            except:
                try:
                    # Try Linux fonts
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
                    label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
                    try:
                        bold_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
                    except:
                        bold_font = label_font
                except:
                    try:
                        # Try Windows fonts
                        font = ImageFont.truetype("arial.ttf", 14)
                        label_font = ImageFont.truetype("arial.ttf", 13)
                        try:
                            bold_font = ImageFont.truetype("arialbd.ttf", 13)
                        except:
                            bold_font = label_font
                    except:
                        # Fallback to default
                        font = ImageFont.load_default()
                        label_font = ImageFont.load_default()
                        bold_font = label_font
            
            # Draw form fields if available (preferred method)
            drawn_boxes = []
            form_fields_data = []
            
            if form_fields:
                for field_info in form_fields:
                    bbox = field_info.get('bounding_box', {})
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
                    
                    # Get field type and colors
                    field_type = field_info.get('field_type', 'unknown')
                    outline_color, fill_color = self._get_field_color(field_type)
                    
                    # Draw semi-transparent fill rectangle
                    # Create a temporary image for transparency
                    overlay = Image.new('RGBA', annotated_image.size, (0, 0, 0, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    overlay_draw.rectangle(
                        [(x1, y1), (x2, y2)],
                        fill=fill_color + (128,),  # 50% opacity
                        outline=None
                    )
                    annotated_image = Image.alpha_composite(annotated_image.convert('RGBA'), overlay).convert('RGB')
                    draw = ImageDraw.Draw(annotated_image)
                    
                    # Draw border with field-specific color (thicker, more visible)
                    draw.rectangle(
                        [(x1, y1), (x2, y2)],
                        outline=outline_color,
                        width=3
                    )
                    
                    # Draw field type label with better styling
                    # Prefer actual label text from the form, fallback to field_type if empty
                    actual_label = field_info.get('label_text', '').strip()
                    if actual_label:
                        # Use the actual label text from the form (truncate if too long)
                        label_text = actual_label[:40] + ('...' if len(actual_label) > 40 else '')
                    else:
                        # Fallback to field type if no label text
                        label_text = field_type.replace('_', ' ').title()  # Convert snake_case to Title Case
                    if label_font:
                        # Calculate text size and position
                        text_bbox = draw.textbbox((0, 0), label_text, font=bold_font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        # Position label above the box with padding
                        label_padding = 4
                        label_x = max(0, x1)  # Ensure label doesn't go off left edge
                        label_y = max(0, y1 - text_height - label_padding * 2)
                        
                        # Draw label background with rounded corners effect (using padding)
                        label_bg_x1 = label_x - label_padding
                        label_bg_y1 = label_y - label_padding
                        label_bg_x2 = label_x + text_width + label_padding
                        label_bg_y2 = label_y + text_height + label_padding
                        
                        draw.rectangle(
                            [(label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2)],
                            fill=outline_color,
                            outline=outline_color
                        )
                        
                        # Draw label text in white
                        draw.text(
                            (label_x, label_y),
                            label_text,
                            fill='white',
                            font=bold_font
                        )
                    
                    # Store form field information
                    form_fields_data.append({
                        'field_type': field_type,
                        'label_text': field_info.get('label_text', ''),
                        'value_text': field_info.get('value_text', ''),
                        'confidence': field_info.get('confidence', 0),
                        'field_confidence': field_info.get('field_confidence', 0),
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
                    
                    # Also add to drawn_boxes for backward compatibility
                    drawn_boxes.append({
                        'block_type': 'FORM_FIELD',
                        'text': f"{field_type}: {field_info.get('label_text', '')}",
                        'confidence': field_info.get('confidence', 0),
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
            
            # Fallback to legacy bounding boxes if no form fields
            # Also classify these boxes if they have text
            elif bounding_boxes:
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
                    
                    # Get text and classify field type if possible
                    text = bbox_info.get('text', '')
                    field_type = 'unknown'
                    field_confidence = 0.0
                    
                    if text:
                        # Try to classify the text to get a field type
                        field_type, field_confidence = self.field_classifier.classify_field(text)
                    
                    # Get colors based on classified field type
                    outline_color, fill_color = self._get_field_color(field_type)
                    
                    # Draw semi-transparent fill rectangle
                    overlay = Image.new('RGBA', annotated_image.size, (0, 0, 0, 0))
                    overlay_draw = ImageDraw.Draw(overlay)
                    overlay_draw.rectangle(
                        [(x1, y1), (x2, y2)],
                        fill=fill_color + (128,),  # 50% opacity
                        outline=None
                    )
                    annotated_image = Image.alpha_composite(annotated_image.convert('RGBA'), overlay).convert('RGB')
                    draw = ImageDraw.Draw(annotated_image)
                    
                    # Draw border with field-specific color
                    draw.rectangle(
                        [(x1, y1), (x2, y2)],
                        outline=outline_color,
                        width=3
                    )
                    
                    # Draw label if available
                    if text and font:
                        # Always use the actual text from the form as the label (truncate if needed)
                        label_text = text[:40] + ('...' if len(text) > 40 else '')
                        
                        # Calculate text size and position
                        text_bbox = draw.textbbox((0, 0), label_text, font=bold_font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        
                        # Position label above the box with padding
                        label_padding = 4
                        label_x = max(0, x1)
                        label_y = max(0, y1 - text_height - label_padding * 2)
                        
                        # Draw label background
                        label_bg_x1 = label_x - label_padding
                        label_bg_y1 = label_y - label_padding
                        label_bg_x2 = label_x + text_width + label_padding
                        label_bg_y2 = label_y + text_height + label_padding
                        
                        draw.rectangle(
                            [(label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2)],
                            fill=outline_color,
                            outline=outline_color
                        )
                        
                        # Draw label text
                        draw.text(
                            (label_x, label_y),
                            label_text,
                            fill='white',
                            font=bold_font
                        )
                    
                    # Store box information with field type
                    drawn_boxes.append({
                        'block_type': bbox_info.get('block_type', ''),
                        'text': text,
                        'field_type': field_type,
                        'field_confidence': field_confidence,
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
                    
                    # Also add to form_fields_data if we classified it
                    if field_type != 'unknown':
                        form_fields_data.append({
                            'field_type': field_type,
                            'label_text': text,
                            'value_text': '',
                            'confidence': bbox_info.get('confidence', 0),
                            'field_confidence': field_confidence,
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
                'boxes': drawn_boxes,  # Legacy format
                'form_fields': form_fields_data,  # New format with field types
                'total_boxes': len(drawn_boxes),
                'total_form_fields': len(form_fields_data),
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

