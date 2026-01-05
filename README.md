# Court Forms OCR Workflow

A full-stack application for scraping PDF forms from California Courts website, processing them with AWS Textract OCR, and displaying results in a modern web interface.

## Features

- **Web Scraping**: Uses Firecrawl to scrape the court forms website and extract PDF links
- **OCR Processing**: Processes PDFs with AWS Textract to extract text and identify form fields
- **Image Annotation**: Creates annotated images showing OCR bounding boxes for the first 5 forms
- **Rate Limiting**: Enforces a 50 call limit across Firecrawl and AWS Textract to control costs
- **Modern Frontend**: Next.js 14 with Material-UI 5, TypeScript, and Tailwind CSS
- **In-Memory Processing**: PDFs are processed in memory without saving to disk

## Project Structure

```
forms_workflow_demo/
├── backend/          # Python FastAPI backend
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Configuration
│   │   ├── models.py            # Pydantic models
│   │   ├── services/            # Business logic services
│   │   ├── utils/               # Utility functions
│   │   └── routes/              # API routes
│   └── requirements.txt
└── frontend/        # Next.js frontend
    ├── app/                     # Next.js app directory
    ├── components/              # React components
    └── lib/                     # API client
```

## Prerequisites

- Python 3.8+
- Node.js 18+
- AWS Account with Textract access
- Firecrawl API key
- poppler-utils (for PDF to image conversion)

### Installing poppler-utils

**macOS:**
```bash
brew install poppler
```

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**Windows:**
Download from: https://github.com/oschwartz10612/poppler-windows/releases

## Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the `backend` directory:
```env
# AWS Configuration
AWS_PROFILE=your-profile
# OR
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_SESSION_TOKEN=your-session-token  # If using temporary credentials
AWS_REGION=us-east-1

# Firecrawl Configuration
FIRECRAWL_API_KEY=your-firecrawl-api-key

# Rate Limiting
MAX_TOTAL_CALLS=50
ENABLE_RATE_LIMITING=true

# Image Processing
SAMPLE_IMAGES_COUNT=5
IMAGES_OUTPUT_DIR=output/images

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000
```

5. Run the backend server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

## Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Create a `.env.local` file:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

4. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

1. Start both the backend and frontend servers
2. Open the frontend in your browser
3. Click "Scrape Forms" to start the scraping process
4. The application will:
   - Scrape the court forms website
   - Extract PDF links
   - Process PDFs with AWS Textract (up to the rate limit)
   - Create annotated images for the first 5 forms
   - Display results in the dashboard

5. Click on a form to view detailed OCR results and annotated images

## API Endpoints

- `GET /api/health` - Health check
- `POST /api/forms/scrape` - Start scraping process
- `GET /api/forms` - List all scraped forms
- `GET /api/forms/{id}` - Get form details with OCR results
- `GET /api/forms/{id}/images` - Get annotated images for a form
- `GET /api/rate-limit` - Get rate limit status

## Rate Limiting

The application enforces a combined rate limit of 50 calls across Firecrawl and AWS Textract. This helps control costs and prevent excessive API usage. The rate limit status is displayed in the frontend dashboard.

## Output

- **OCR Results**: Markdown text extracted from PDFs
- **Annotated Images**: PNG images with OCR bounding boxes drawn (saved in `output/images/`)
- **JSON Metadata**: Bounding box coordinates and text for each annotated image

## Troubleshooting

### AWS Credentials Issues
- Ensure your AWS credentials are properly configured
- If using temporary credentials, make sure `AWS_SESSION_TOKEN` is set
- Check that your AWS account has Textract permissions

### PDF to Image Conversion Issues
- Ensure poppler-utils is installed and in your PATH
- On macOS, you may need to set the `PATH` environment variable

### Rate Limit Reached
- The application will stop processing when the 50 call limit is reached
- Check the rate limit status in the dashboard
- Adjust `MAX_TOTAL_CALLS` in the `.env` file if needed

## License

MIT
