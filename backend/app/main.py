"""
FastAPI application for Court Forms OCR Workflow.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from datetime import datetime
from typing import List, Optional
import logging
import uuid
import json
import asyncio
import threading
from pathlib import Path

from app.config import Config
from app.models import (
    HealthResponse, ScrapeRequest, ScrapeResponse,
    FormInfo, FormDetail, RateLimitStatus, ImageMetadata
)
from app.utils.rate_limiter import RateLimiter
from app.services.firecrawl_service import FirecrawlService
from app.services.textract_service import TextractService
from app.services.image_processor import ImageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Court Forms OCR Workflow API",
    description="API for scraping and OCR processing of court forms",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize rate limiter (doesn't require API keys)
rate_limiter = RateLimiter(max_total_calls=Config.MAX_TOTAL_CALLS)

# Services will be initialized lazily when needed (in startup or first use)
firecrawl_service: Optional[FirecrawlService] = None
textract_service: Optional[TextractService] = None
image_processor: Optional[ImageProcessor] = None

# In-memory storage for forms (in production, use a database)
forms_storage: dict = {}
images_storage: dict = {}

# Store for SSE connections (job_id -> list of messages)
# Background thread appends messages, SSE endpoint reads them
sse_messages: dict = {}
sse_message_locks: dict = {}


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    global firecrawl_service, textract_service, image_processor
    
    try:
        Config.validate()
        logger.info("Configuration validated successfully")
        
        # Initialize services after config validation
        firecrawl_service = FirecrawlService(rate_limiter=rate_limiter)
        textract_service = TextractService(rate_limiter=rate_limiter)
        image_processor = ImageProcessor()
        logger.info("Services initialized successfully")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now()
    )


@app.post("/api/forms/scrape", response_model=ScrapeResponse)
async def scrape_forms(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """
    Start scraping forms from the court forms website.
    This endpoint initiates the scraping process in the background.
    """
    job_id = str(uuid.uuid4())
    
    # Initialize SSE message storage for this job
    sse_messages[job_id] = []
    sse_message_locks[job_id] = threading.Lock()
    
    # Start background task
    background_tasks.add_task(
        process_forms_scraping,
        url=request.url,
        job_id=job_id
    )
    
    return ScrapeResponse(
        message="Scraping initiated",
        job_id=job_id,
        estimated_forms=None
    )


@app.get("/api/forms/scrape/{job_id}/stream")
async def stream_scrape_progress(job_id: str):
    """
    Server-Sent Events endpoint for real-time scraping progress updates.
    """
    async def event_generator():
        if job_id not in sse_messages:
            # Initialize if doesn't exist
            sse_messages[job_id] = []
            sse_message_locks[job_id] = threading.Lock()
        
        last_index = 0
        
        try:
            while True:
                # Get new messages since last check
                with sse_message_locks.get(job_id, threading.Lock()):
                    messages = sse_messages.get(job_id, [])
                    new_messages = messages[last_index:]
                    last_index = len(messages)
                
                # Send all new messages
                for message in new_messages:
                    yield message
                    
                    # Check if it's a completion message
                    if 'event: complete' in message or 'event: error' in message:
                        # Wait a bit then cleanup
                        await asyncio.sleep(1)
                        return
                
                # If no new messages, wait a bit
                if not new_messages:
                    await asyncio.sleep(0.5)
                    # Send keepalive every 30 seconds
                    yield f"event: keepalive\ndata: {json.dumps({'timestamp': datetime.now().isoformat()})}\n\n"
                    
        except Exception as e:
            logger.error(f"Error in SSE stream for job {job_id}: {e}")
        finally:
            # Cleanup after delay
            await asyncio.sleep(2)
            sse_messages.pop(job_id, None)
            sse_message_locks.pop(job_id, None)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/forms", response_model=List[FormInfo])
async def list_forms():
    """List all scraped forms."""
    return list(forms_storage.values())


@app.get("/api/forms/{form_id}", response_model=FormDetail)
async def get_form(form_id: str):
    """Get detailed information about a specific form."""
    if form_id not in forms_storage:
        raise HTTPException(status_code=404, detail="Form not found")
    return forms_storage[form_id]


@app.get("/api/forms/{form_id}/images", response_model=List[ImageMetadata])
async def get_form_images(form_id: str):
    """Get annotated images for a specific form."""
    if form_id not in images_storage:
        return []
    return images_storage[form_id]


@app.get("/api/images/{image_filename:path}")
async def serve_image(image_filename: str):
    """Serve annotated images."""
    from app.config import Config
    image_path = Path(Config.IMAGES_OUTPUT_DIR) / image_filename
    
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(image_path)


@app.get("/api/rate-limit", response_model=RateLimitStatus)
async def get_rate_limit_status():
    """Get current rate limit status."""
    stats = rate_limiter.get_stats()
    return RateLimitStatus(
        total_calls=stats['total_calls'],
        max_calls=stats['max_calls'],
        remaining_calls=stats['remaining_calls'],
        calls_by_service=stats['calls_by_service']
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


def _send_sse_update(job_id: str, event_type: str, data: dict):
    """Send SSE update to connected clients."""
    if job_id in sse_messages:
        try:
            message = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            with sse_message_locks.get(job_id, threading.Lock()):
                if job_id not in sse_messages:
                    sse_messages[job_id] = []
                sse_messages[job_id].append(message)
        except Exception as e:
            logger.warning(f"Failed to send SSE update for job {job_id}: {e}")


def process_forms_scraping(url: str, job_id: str):
    """
    Background task to scrape forms, process with OCR, and create annotated images.
    Processes documents sequentially: download -> textract -> save -> image -> next
    
    Args:
        url: URL to scrape
        job_id: Job identifier
    """
    global firecrawl_service, textract_service, image_processor
    from app.utils.pdf_handler import PDFHandler
    
    try:
        logger.info(f"Starting forms scraping job {job_id} for URL: {url}")
        _send_sse_update(job_id, 'start', {'job_id': job_id, 'message': 'Starting scraping...'})
        
        # Ensure services are initialized
        if firecrawl_service is None:
            firecrawl_service = FirecrawlService(rate_limiter=rate_limiter)
        if textract_service is None:
            textract_service = TextractService(rate_limiter=rate_limiter)
        if image_processor is None:
            image_processor = ImageProcessor()
        
        pdf_handler = PDFHandler()
        
        # Step 1: Scrape page and extract PDF URLs (don't download yet)
        forms_url = 'https://selfhelp.courts.ca.gov/find-forms/all'
        pdf_infos = firecrawl_service.scrape_and_extract_pdfs(forms_url)
        logger.info(f"Found {len(pdf_infos)} PDFs to process")
        _send_sse_update(job_id, 'progress', {
            'total': len(pdf_infos),
            'processed': 0,
            'message': f'Found {len(pdf_infos)} PDFs to process'
        })
        
        if not pdf_infos:
            logger.warning("No PDFs found to process")
            _send_sse_update(job_id, 'complete', {'message': 'No PDFs found'})
            return
        
        # Step 2: Process each PDF sequentially: download -> textract -> save -> image -> next
        processed_count = 0
        images_created = 0
        
        for idx, pdf_info in enumerate(pdf_infos, 1):
            # Check rate limit before processing
            can_call, reason = rate_limiter.can_make_call('textract')
            if not can_call:
                logger.warning(f"Rate limit reached. Stopping processing. Reason: {reason}")
                _send_sse_update(job_id, 'rate_limit', {'message': f'Rate limit reached: {reason}'})
                break
            
            form_id = str(uuid.uuid4())
            form_name = pdf_info.get('name', 'Unknown Form')
            pdf_url = pdf_info.get('url', '')
            
            # Create form entry with pending status
            form_entry = FormDetail(
                id=form_id,
                name=form_name,
                url=pdf_url,
                status='downloading',
                created_at=datetime.now()
            )
            forms_storage[form_id] = form_entry
            _send_sse_update(job_id, 'form_start', {
                'form_id': form_id,
                'form_name': form_name,
                'index': idx,
                'total': len(pdf_infos)
            })
            
            try:
                # Download PDF
                logger.info(f"[{idx}/{len(pdf_infos)}] Downloading: {form_name}")
                form_entry.status = 'downloading'
                pdf_bytes = pdf_handler.download_pdf(pdf_url)
                
                if not pdf_bytes:
                    raise Exception(f"Failed to download PDF from {pdf_url}")
                
                logger.info(f"[{idx}/{len(pdf_infos)}] Downloaded: {form_name} ({len(pdf_bytes)} bytes)")
                
                # Process with Textract
                form_entry.status = 'processing'
                _send_sse_update(job_id, 'form_processing', {
                    'form_id': form_id,
                    'form_name': form_name,
                    'message': 'Processing with Textract...'
                })
                
                ocr_result = textract_service.process_pdf(pdf_bytes)
                
                if ocr_result.get('success'):
                    # Update form entry with results
                    form_entry.status = 'completed'
                    form_entry.markdown_text = ocr_result.get('markdown', '')
                    form_entry.ocr_confidence = ocr_result.get('metadata', {}).get('confidence', 0)
                    form_entry.text_length = len(ocr_result.get('text', ''))
                    form_entry.blocks_count = ocr_result.get('metadata', {}).get('blocks_count', 0)
                    
                    processed_count += 1
                    logger.info(f"[{idx}/{len(pdf_infos)}] Processed: {form_name}")
                    
                    # Create annotated image if we haven't created 5 yet
                    if images_created < Config.SAMPLE_IMAGES_COUNT:
                        form_entry.status = 'creating_image'
                        _send_sse_update(job_id, 'form_image', {
                            'form_id': form_id,
                            'form_name': form_name,
                            'message': 'Creating annotated image...'
                        })
                        
                        image_result = image_processor.create_annotated_image(
                            pdf_bytes=pdf_bytes,
                            bounding_boxes=ocr_result.get('bounding_boxes', []),
                            form_id=form_id,
                            form_name=form_name
                        )
                        
                        if image_result:
                            if form_id not in images_storage:
                                images_storage[form_id] = []
                            
                            images_storage[form_id].append(
                                ImageMetadata(
                                    image_path=image_result['image_path'],
                                    form_id=form_id,
                                    form_name=form_name,
                                    boxes=image_result['metadata']['boxes'],
                                    created_at=datetime.fromisoformat(image_result['metadata']['created_at'])
                                )
                            )
                            images_created += 1
                            logger.info(f"Created annotated image for form {form_id}: {image_result['image_path']}")
                        
                        form_entry.status = 'completed'
                    
                    _send_sse_update(job_id, 'form_complete', {
                        'form_id': form_id,
                        'form_name': form_name,
                        'processed': processed_count,
                        'total': len(pdf_infos)
                    })
                else:
                    form_entry.status = 'error'
                    form_entry.error_message = ocr_result.get('error', 'Unknown error')
                    logger.error(f"Failed to process form {form_id}: {form_entry.error_message}")
                    _send_sse_update(job_id, 'form_error', {
                        'form_id': form_id,
                        'form_name': form_name,
                        'error': form_entry.error_message
                    })
                    
            except Exception as e:
                logger.error(f"Error processing form {form_id}: {e}")
                form_entry.status = 'error'
                form_entry.error_message = str(e)
                _send_sse_update(job_id, 'form_error', {
                    'form_id': form_id,
                    'form_name': form_name,
                    'error': str(e)
                })
            
            # Send overall progress update
            _send_sse_update(job_id, 'progress', {
                'total': len(pdf_infos),
                'processed': processed_count,
                'current': idx,
                'message': f'Processed {processed_count}/{len(pdf_infos)} forms'
            })
        
        # Cleanup
        logger.info(f"Completed scraping job {job_id}. Processed {processed_count} forms, created {images_created} annotated images")
        _send_sse_update(job_id, 'complete', {
            'processed': processed_count,
            'total': len(pdf_infos),
            'images_created': images_created,
            'message': f'Completed: {processed_count} forms processed, {images_created} images created'
        })
        
    except Exception as e:
        logger.error(f"Error in scraping job {job_id}: {e}", exc_info=True)
        _send_sse_update(job_id, 'error', {'message': str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True
    )

