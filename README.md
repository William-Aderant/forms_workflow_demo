# Court Forms OCR Workflow

A full-stack application for scraping PDF forms from California Courts website, processing them with AWS Textract OCR, and displaying results in a modern web interface. Features a technology stack comparison pipeline that shows the incremental value of each AI component.

## Features

- **Web Scraping**: Uses Firecrawl to scrape the court forms website and extract PDF links
- **OCR Processing**: Processes PDFs with AWS Textract to extract text and identify form fields
- **Image Annotation**: Creates annotated images showing OCR bounding boxes for ALL forms
- **Rate Limiting**: Enforces a 50 call limit across Firecrawl and AWS Textract to control costs
- **Modern Frontend**: Next.js 14 with Material-UI 5, TypeScript, and Tailwind CSS
- **In-Memory Processing**: PDFs are processed in memory without saving to disk

### 🆕 Legal Form Pipeline & Technology Comparison

- **LayoutLMv3 Integration**: Layout-aware semantic classification using transformer models
- **GLM-4.5V Adjudication**: Vision-language model for ambiguous field resolution
- **Technology Stack Comparison**: Side-by-side comparison showing results across 4 configurations:
  1. **Textract Only**: Baseline OCR detection
  2. **Textract + Heuristics**: Rule-based pattern matching
  3. **Textract + LayoutLM**: Layout-aware ML classification
  4. **Full Stack**: Complete pipeline with GLM adjudication
- **Semantic Labeling**: 55+ legal-specific field labels (PERSON_NAME, CASE_NUMBER, SIGNATURE, etc.)
- **Audit Trail**: Full logging of all classifications and adjudications for legal review

## Project Structure

```
forms_workflow_demo/
├── backend/          # Python FastAPI backend
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── config.py            # Configuration
│   │   ├── models.py            # Pydantic models
│   │   ├── services/            # Business logic services
│   │   │   └── legal_form_pipeline/  # 🆕 Legal form understanding pipeline
│   │   │       ├── ontology.py       # Semantic label definitions
│   │   │       ├── field_candidates.py # Field detection
│   │   │       ├── layoutlm_classifier.py # LayoutLMv3 classification
│   │   │       ├── glm_adjudicator.py # GLM-4.5V adjudication
│   │   │       ├── pipeline.py       # Main orchestrator
│   │   │       └── comparison_pipeline.py # 🆕 Technology comparison
│   │   ├── utils/               # Utility functions
│   │   └── routes/              # API routes
│   │       └── legal_form_pipeline.py # 🆕 Comparison API endpoints
│   └── requirements.txt
└── frontend/        # Next.js frontend
    ├── app/                     # Next.js app directory
    ├── components/              # React components
    │   └── TechStackComparison.tsx # 🆕 Comparison UI component
    └── lib/                     # API client
```

## Prerequisites

- Python 3.8+
- Node.js 18+
- AWS Account with Textract access
- Firecrawl API key
- poppler-utils (for PDF to image conversion)
- Optional: GPU for faster LayoutLM inference
- Optional: GLM API key for VLM adjudication

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

# GLM-4.5V Configuration (optional)
GLM_API_KEY=your-glm-api-key
# OR
ZHIPUAI_API_KEY=your-zhipuai-api-key

# Rate Limiting
MAX_TOTAL_CALLS=50
ENABLE_RATE_LIMITING=true

# Image Processing (0 = save ALL images)
SAMPLE_IMAGES_COUNT=0
IMAGES_OUTPUT_DIR=output/images

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000
```

5. Run the backend server:
```bash
uvicorn app.main:app --reload --port 8000
```

Alternatively, you can run it using Python which will use the config:
```bash
python -m app.main
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

### Form Scraping

1. Start both the backend and frontend servers
2. Open the frontend in your browser
3. Click "Scrape Forms" to start the scraping process
4. The application will:
   - Scrape the court forms website
   - Extract PDF links
   - Process PDFs with AWS Textract (up to the rate limit)
   - Create annotated images for ALL forms
   - Display results in the dashboard

5. Click on a form to view detailed OCR results and annotated images

### 🆕 Technology Stack Comparison

1. Navigate to the "Technology Stack Comparison" tab
2. Upload a PDF form
3. Click "Compare Technologies"
4. View side-by-side results showing:
   - Classification accuracy at each technology level
   - Confidence improvements
   - Field-by-field comparison
   - Processing time for each configuration

This helps demonstrate the value of each component:
- How much LayoutLM improves over simple heuristics
- When GLM adjudication helps resolve ambiguous cases
- Overall accuracy gains from the full stack

## API Endpoints

### Core APIs
- `GET /api/health` - Health check
- `POST /api/forms/scrape` - Start scraping process
- `GET /api/forms` - List all scraped forms
- `GET /api/forms/{id}` - Get form details with OCR results
- `GET /api/forms/{id}/images` - Get annotated images for a form
- `GET /api/rate-limit` - Get rate limit status

### 🆕 Legal Form Pipeline APIs
- `POST /api/v1/legal-forms/analyze` - Analyze a PDF for empty fields
- `POST /api/v1/legal-forms/analyze/extended` - Analysis with full audit trail
- `POST /api/v1/legal-forms/compare` - **Compare across all technology configurations**
- `POST /api/v1/legal-forms/compare-textract` - Compare using pre-extracted Textract
- `GET /api/v1/legal-forms/comparison-modes` - Get available comparison modes
- `GET /api/v1/legal-forms/ontology` - Get semantic label schema
- `GET /api/v1/legal-forms/health` - Pipeline component status

## Technology Comparison Modes

| Mode | Components | Description |
|------|------------|-------------|
| `textract_only` | AWS Textract | Baseline OCR - detects fields but no classification |
| `textract_heuristics` | Textract + Rules | Pattern matching for common field labels |
| `textract_layoutlm` | Textract + LayoutLMv3 | Layout-aware ML classification |
| `full_stack` | Textract + LayoutLM + GLM | Complete pipeline with VLM adjudication |

## Semantic Label Ontology

The pipeline supports 55+ legal-specific labels:

- **Identity**: `PERSON_NAME`, `ATTORNEY_NAME`, `LAW_FIRM_NAME`, `PLAINTIFF_NAME`, `DEFENDANT_NAME`
- **Case**: `CASE_NUMBER`, `DOCKET_NUMBER`, `COURT_NAME`, `JUDICIAL_DISTRICT`
- **Date/Time**: `DATE`, `DATE_OF_BIRTH`, `FILING_DATE`, `HEARING_DATE`
- **Contact**: `ADDRESS`, `PHONE`, `EMAIL`, `FAX`
- **Signature**: `SIGNATURE`, `INITIALS`, `NOTARY_SIGNATURE`
- **Attestation**: `AGE_OVER_18_CONFIRMATION`, `UNDER_PENALTY_OF_PERJURY`
- **Checkbox**: `CHECKBOX_PRO_SE`, `CHECKBOX_FEE_WAIVER`, `CHECKBOX_INTERPRETER_NEEDED`

## Rate Limiting

The application enforces a combined rate limit of 50 calls across Firecrawl and AWS Textract. This helps control costs and prevent excessive API usage. The rate limit status is displayed in the frontend dashboard.

## Output

- **OCR Results**: Markdown text extracted from PDFs
- **Annotated Images**: PNG images with OCR bounding boxes drawn (saved in `output/images/`)
- **JSON Metadata**: Bounding box coordinates and text for each annotated image
- **Comparison Results**: JSON showing field classifications across all technology configurations

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

### LayoutLM Issues
- If transformers library is not installed, the pipeline falls back to heuristics
- GPU is recommended for faster inference but not required
- The pipeline will automatically use CPU if no GPU is available

### GLM Adjudication Issues
- If `GLM_API_KEY` is not set, the adjudicator uses heuristic fallback
- This means the comparison will still work, just with simulated GLM behavior

## License

MIT
