#!/usr/bin/env python3
"""
Legal Form Pipeline - Example Usage
====================================

This script demonstrates how to use the Legal Form Understanding Pipeline
for detecting and classifying empty fillable fields in court forms.

Usage:
    python examples/legal_form_pipeline_example.py path/to/form.pdf

Requirements:
    - AWS credentials configured (for Textract)
    - Optional: GLM_API_KEY environment variable (for adjudication)
    - Optional: GPU for faster LayoutLM inference
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.legal_form_pipeline import (
    LegalFormPipeline,
    LegalFieldOntology,
    SemanticLabel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_pdf(pdf_path: str, output_path: str = None, extended: bool = False):
    """
    Process a PDF through the legal form pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save JSON output
        extended: Whether to include extended metadata in output
    """
    # Read PDF
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"File not found: {pdf_path}")
        return None
    
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    logger.info(f"Processing: {pdf_path.name} ({len(pdf_bytes):,} bytes)")
    
    # Initialize pipeline
    pipeline = LegalFormPipeline(
        use_gpu=True,  # Will fall back to CPU if unavailable
        enable_glm_adjudication=True  # Enable for ambiguous cases
    )
    
    logger.info(f"Pipeline mode: LayoutLM={pipeline.classifier.mode}")
    
    # Process the document
    result = pipeline.process_pdf(pdf_bytes, document_id=pdf_path.stem)
    
    # Get statistics
    stats = pipeline.get_statistics(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("LEGAL FORM PIPELINE RESULTS")
    print("=" * 60)
    print(f"\nDocument ID: {result.document_id}")
    print(f"Total Pages: {result.total_pages}")
    print(f"Total Fields Detected: {result.total_fields}")
    print(f"Fields Adjudicated: {result.total_adjudicated}")
    print(f"Processing Time: {result.total_processing_time_ms}ms")
    
    # Print field summary
    print("\n" + "-" * 40)
    print("DETECTED FIELDS")
    print("-" * 40)
    
    for page in result.pages:
        print(f"\nPage {page.page}: {len(page.fields)} fields")
        
        for field in page.fields:
            conf_indicator = "✓" if field.confidence >= 0.7 else "?" if field.confidence >= 0.4 else "✗"
            adj_indicator = " [GLM]" if field.was_adjudicated else ""
            
            print(f"  {conf_indicator} [{field.field_type:10}] {field.semantic_label:30} "
                  f"(conf: {field.confidence:.2f}){adj_indicator}")
            if field.supporting_text:
                print(f"    └─ Label: \"{field.supporting_text[:50]}{'...' if len(field.supporting_text) > 50 else ''}\"")
    
    # Print statistics
    print("\n" + "-" * 40)
    print("STATISTICS")
    print("-" * 40)
    print(f"Average Confidence: {stats['average_confidence']:.2%}")
    print(f"Unknown Field Rate: {stats['unknown_field_rate']:.1%}")
    print(f"Adjudication Rate: {stats['adjudication_rate']:.1%}")
    
    print("\nLabel Distribution:")
    for label, count in sorted(stats['label_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")
    
    print("\nField Type Distribution:")
    for ftype, count in stats['field_type_distribution'].items():
        print(f"  {ftype}: {count}")
    
    # Save output if path provided
    if output_path:
        output_path = Path(output_path)
        
        if extended:
            output_data = result.to_extended_dict()
        else:
            output_data = result.to_dict()
        
        output_data['statistics'] = stats
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Output saved to: {output_path}")
    
    return result


def show_ontology():
    """Print the complete semantic label ontology."""
    ontology = LegalFieldOntology()
    
    print("\n" + "=" * 60)
    print("SEMANTIC LABEL ONTOLOGY")
    print("=" * 60)
    print(f"\nTotal Labels: {ontology.num_labels}")
    
    # Group by category
    categories = {
        'Identity': ['NAME', 'ATTORNEY', 'PARTY', 'PLAINTIFF', 'DEFENDANT', 
                    'PETITIONER', 'RESPONDENT', 'WITNESS', 'JUDGE', 'FIRM'],
        'Case': ['CASE', 'DOCKET', 'COURT', 'JUDICIAL', 'COUNTY'],
        'Date/Time': ['DATE', 'TIME'],
        'Contact': ['ADDRESS', 'STREET', 'CITY', 'STATE', 'ZIP', 'PHONE', 'FAX', 'EMAIL'],
        'Signature': ['SIGNATURE', 'INITIALS', 'NOTARY'],
        'Attestation': ['AGE_OVER_18', 'PERJURY', 'DECLARATION', 'CHECKBOX'],
        'Monetary': ['AMOUNT', 'FEE', 'BOND', 'DAMAGES'],
        'Narrative': ['DESCRIPTION', 'CAUSE', 'RELIEF', 'FACTS', 'ADDITIONAL'],
        'Document': ['EXHIBIT', 'PAGE', 'ATTACHMENT'],
    }
    
    uncategorized = set(l.value for l in SemanticLabel)
    
    for category, keywords in categories.items():
        matching = [l for l in SemanticLabel 
                   if any(kw in l.value for kw in keywords)]
        
        if matching:
            print(f"\n{category}:")
            for label in matching:
                meta = ontology.get_metadata(label)
                attestation = " ⚖️" if meta and meta.is_attestational else ""
                print(f"  • {label.value}{attestation}")
                uncategorized.discard(label.value)
    
    if 'UNKNOWN_FIELD' in uncategorized:
        uncategorized.remove('UNKNOWN_FIELD')
        print(f"\nFallback:")
        print(f"  • UNKNOWN_FIELD")
    
    print("\n⚖️ = Attestational (legal significance, extra scrutiny)")


def main():
    parser = argparse.ArgumentParser(
        description='Legal Form Understanding Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a PDF and print results
    python legal_form_pipeline_example.py form.pdf
    
    # Process and save output to JSON
    python legal_form_pipeline_example.py form.pdf -o results.json
    
    # Save extended output with audit trail
    python legal_form_pipeline_example.py form.pdf -o results.json --extended
    
    # Show the semantic label ontology
    python legal_form_pipeline_example.py --ontology
        """
    )
    
    parser.add_argument(
        'pdf_path',
        nargs='?',
        help='Path to PDF file to process'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Path to save JSON output'
    )
    
    parser.add_argument(
        '--extended',
        action='store_true',
        help='Include extended metadata and audit trail in output'
    )
    
    parser.add_argument(
        '--ontology',
        action='store_true',
        help='Show the semantic label ontology and exit'
    )
    
    args = parser.parse_args()
    
    if args.ontology:
        show_ontology()
        return
    
    if not args.pdf_path:
        parser.print_help()
        print("\nError: Please provide a PDF path or use --ontology")
        sys.exit(1)
    
    process_pdf(args.pdf_path, args.output, args.extended)


if __name__ == '__main__':
    main()

