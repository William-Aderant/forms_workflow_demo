"""
Legal Field Ontology
====================

Defines the canonical set of semantic labels for legal form fields.
This ontology is CLOSED - no labels outside this set are permitted.

Design Rationale:
-----------------
The ontology balances granularity with practical utility:
- Too granular: classifier struggles with fine distinctions
- Too coarse: loses semantic meaning for downstream applications

Labels are derived from analysis of common court form patterns:
- California Judicial Council forms
- Federal court forms (PACER)
- State-specific civil/criminal forms

IMPORTANT: This is a CLOSED ontology. GLM-4.5V is explicitly forbidden
from inventing new labels. Any field that doesn't match gets UNKNOWN_FIELD.
"""

from enum import Enum
from typing import Dict, List, Set, Optional
from dataclasses import dataclass


class SemanticLabel(str, Enum):
    """
    Canonical semantic labels for legal form fields.
    
    Categories:
    - Identity fields: PERSON_NAME, LAW_FIRM_NAME, etc.
    - Case fields: CASE_NUMBER, COURT_NAME, etc.
    - Date/time fields: DATE, DATE_OF_BIRTH, etc.
    - Contact fields: ADDRESS, PHONE, EMAIL
    - Legal attestation: SIGNATURE, AGE_OVER_18_CONFIRMATION, etc.
    - Monetary: AMOUNT, FEE
    - Narrative: DESCRIPTION, CAUSE_OF_ACTION
    - Unknown: UNKNOWN_FIELD (fallback)
    """
    
    # === IDENTITY FIELDS ===
    PERSON_NAME = "PERSON_NAME"
    FIRST_NAME = "FIRST_NAME"
    MIDDLE_NAME = "MIDDLE_NAME"
    LAST_NAME = "LAST_NAME"
    LAW_FIRM_NAME = "LAW_FIRM_NAME"
    ATTORNEY_NAME = "ATTORNEY_NAME"
    ATTORNEY_BAR_NUMBER = "ATTORNEY_BAR_NUMBER"
    PARTY_NAME = "PARTY_NAME"  # Generic party (plaintiff/defendant)
    PLAINTIFF_NAME = "PLAINTIFF_NAME"
    DEFENDANT_NAME = "DEFENDANT_NAME"
    PETITIONER_NAME = "PETITIONER_NAME"
    RESPONDENT_NAME = "RESPONDENT_NAME"
    WITNESS_NAME = "WITNESS_NAME"
    JUDGE_NAME = "JUDGE_NAME"
    
    # === CASE IDENTIFICATION ===
    CASE_NUMBER = "CASE_NUMBER"
    DOCKET_NUMBER = "DOCKET_NUMBER"
    COURT_NAME = "COURT_NAME"
    COURT_DEPARTMENT = "COURT_DEPARTMENT"
    COURT_DIVISION = "COURT_DIVISION"
    COURT_ADDRESS = "COURT_ADDRESS"
    JUDICIAL_DISTRICT = "JUDICIAL_DISTRICT"
    COUNTY = "COUNTY"
    
    # === DATE/TIME FIELDS ===
    DATE = "DATE"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    FILING_DATE = "FILING_DATE"
    HEARING_DATE = "HEARING_DATE"
    HEARING_TIME = "HEARING_TIME"
    SERVICE_DATE = "SERVICE_DATE"
    INCIDENT_DATE = "INCIDENT_DATE"
    
    # === CONTACT INFORMATION ===
    ADDRESS = "ADDRESS"
    STREET_ADDRESS = "STREET_ADDRESS"
    CITY = "CITY"
    STATE = "STATE"
    ZIP_CODE = "ZIP_CODE"
    PHONE = "PHONE"
    FAX = "FAX"
    EMAIL = "EMAIL"
    
    # === LEGAL ATTESTATION/SIGNATURE ===
    SIGNATURE = "SIGNATURE"
    INITIALS = "INITIALS"
    AGE_OVER_18_CONFIRMATION = "AGE_OVER_18_CONFIRMATION"
    UNDER_PENALTY_OF_PERJURY = "UNDER_PENALTY_OF_PERJURY"
    DECLARATION_CONFIRMATION = "DECLARATION_CONFIRMATION"
    NOTARY_SIGNATURE = "NOTARY_SIGNATURE"
    NOTARY_COMMISSION_EXPIRY = "NOTARY_COMMISSION_EXPIRY"
    
    # === MONETARY FIELDS ===
    AMOUNT = "AMOUNT"
    DAMAGES_AMOUNT = "DAMAGES_AMOUNT"
    FILING_FEE = "FILING_FEE"
    BOND_AMOUNT = "BOND_AMOUNT"
    
    # === NARRATIVE/DESCRIPTIVE ===
    DESCRIPTION = "DESCRIPTION"
    CAUSE_OF_ACTION = "CAUSE_OF_ACTION"
    RELIEF_REQUESTED = "RELIEF_REQUESTED"
    FACTS = "FACTS"
    ADDITIONAL_INFORMATION = "ADDITIONAL_INFORMATION"
    
    # === DOCUMENT REFERENCE ===
    EXHIBIT_NUMBER = "EXHIBIT_NUMBER"
    PAGE_NUMBER = "PAGE_NUMBER"
    ATTACHMENT_NUMBER = "ATTACHMENT_NUMBER"
    
    # === CHECKBOX SPECIFIC (legal attestations) ===
    CHECKBOX_REPRESENTED_BY_ATTORNEY = "CHECKBOX_REPRESENTED_BY_ATTORNEY"
    CHECKBOX_PRO_SE = "CHECKBOX_PRO_SE"
    CHECKBOX_FEE_WAIVER = "CHECKBOX_FEE_WAIVER"
    CHECKBOX_INTERPRETER_NEEDED = "CHECKBOX_INTERPRETER_NEEDED"
    CHECKBOX_DISABILITY_ACCOMMODATION = "CHECKBOX_DISABILITY_ACCOMMODATION"
    CHECKBOX_CONSENT = "CHECKBOX_CONSENT"
    CHECKBOX_CERTIFICATION = "CHECKBOX_CERTIFICATION"
    
    # === FALLBACK ===
    UNKNOWN_FIELD = "UNKNOWN_FIELD"


@dataclass
class LabelMetadata:
    """Metadata for a semantic label including classification hints."""
    label: SemanticLabel
    aliases: List[str]  # Alternative text patterns that map to this label
    context_keywords: List[str]  # Keywords that increase likelihood
    typical_field_types: List[str]  # Expected field types: line, checkbox, table_cell
    is_attestational: bool  # Requires extra scrutiny (legal significance)
    max_expected_length: Optional[int]  # Character limit hint for validation


class LegalFieldOntology:
    """
    Manages the legal field ontology with lookup utilities.
    
    Thread-safe: all data is immutable after initialization.
    """
    
    # Mapping from label to metadata
    # This provides rich context for classification and validation
    _LABEL_METADATA: Dict[SemanticLabel, LabelMetadata] = {
        SemanticLabel.PERSON_NAME: LabelMetadata(
            label=SemanticLabel.PERSON_NAME,
            aliases=["name", "your name", "print name", "printed name", "type or print name", "full name"],
            context_keywords=["name", "print", "type"],
            typical_field_types=["line"],
            is_attestational=False,
            max_expected_length=100
        ),
        SemanticLabel.FIRST_NAME: LabelMetadata(
            label=SemanticLabel.FIRST_NAME,
            aliases=["first name", "given name", "first"],
            context_keywords=["first", "given"],
            typical_field_types=["line"],
            is_attestational=False,
            max_expected_length=50
        ),
        SemanticLabel.LAST_NAME: LabelMetadata(
            label=SemanticLabel.LAST_NAME,
            aliases=["last name", "surname", "family name", "last"],
            context_keywords=["last", "surname", "family"],
            typical_field_types=["line"],
            is_attestational=False,
            max_expected_length=50
        ),
        SemanticLabel.LAW_FIRM_NAME: LabelMetadata(
            label=SemanticLabel.LAW_FIRM_NAME,
            aliases=["firm name", "law firm", "firm"],
            context_keywords=["firm", "law office", "llp", "llc", "pllc", "pc"],
            typical_field_types=["line"],
            is_attestational=False,
            max_expected_length=150
        ),
        SemanticLabel.ATTORNEY_NAME: LabelMetadata(
            label=SemanticLabel.ATTORNEY_NAME,
            aliases=["attorney", "attorney name", "counsel", "attorney for"],
            context_keywords=["attorney", "counsel", "esq", "lawyer"],
            typical_field_types=["line"],
            is_attestational=False,
            max_expected_length=100
        ),
        SemanticLabel.ATTORNEY_BAR_NUMBER: LabelMetadata(
            label=SemanticLabel.ATTORNEY_BAR_NUMBER,
            aliases=["bar number", "bar no", "bar #", "state bar number", "sbn"],
            context_keywords=["bar", "sbn", "license"],
            typical_field_types=["line"],
            is_attestational=False,
            max_expected_length=20
        ),
        SemanticLabel.CASE_NUMBER: LabelMetadata(
            label=SemanticLabel.CASE_NUMBER,
            aliases=["case number", "case no", "case #", "case"],
            context_keywords=["case", "matter", "action"],
            typical_field_types=["line"],
            is_attestational=False,
            max_expected_length=30
        ),
        SemanticLabel.COURT_NAME: LabelMetadata(
            label=SemanticLabel.COURT_NAME,
            aliases=["court", "court name", "superior court", "district court"],
            context_keywords=["court", "superior", "district", "municipal", "circuit"],
            typical_field_types=["line"],
            is_attestational=False,
            max_expected_length=150
        ),
        SemanticLabel.DATE: LabelMetadata(
            label=SemanticLabel.DATE,
            aliases=["date", "dated"],
            context_keywords=["date", "day", "month", "year"],
            typical_field_types=["line"],
            is_attestational=False,
            max_expected_length=20
        ),
        SemanticLabel.SIGNATURE: LabelMetadata(
            label=SemanticLabel.SIGNATURE,
            aliases=["signature", "sign here", "signed", "your signature", "signature of"],
            context_keywords=["sign", "signature", "signed"],
            typical_field_types=["line"],
            is_attestational=True,  # Legal significance!
            max_expected_length=None  # Signature fields are variable
        ),
        SemanticLabel.ADDRESS: LabelMetadata(
            label=SemanticLabel.ADDRESS,
            aliases=["address", "mailing address", "street address", "residence address"],
            context_keywords=["address", "street", "mailing", "residence"],
            typical_field_types=["line"],
            is_attestational=False,
            max_expected_length=200
        ),
        SemanticLabel.PHONE: LabelMetadata(
            label=SemanticLabel.PHONE,
            aliases=["phone", "telephone", "phone number", "tel", "contact number"],
            context_keywords=["phone", "telephone", "tel", "call"],
            typical_field_types=["line"],
            is_attestational=False,
            max_expected_length=20
        ),
        SemanticLabel.EMAIL: LabelMetadata(
            label=SemanticLabel.EMAIL,
            aliases=["email", "e-mail", "email address"],
            context_keywords=["email", "e-mail", "@"],
            typical_field_types=["line"],
            is_attestational=False,
            max_expected_length=100
        ),
        SemanticLabel.AGE_OVER_18_CONFIRMATION: LabelMetadata(
            label=SemanticLabel.AGE_OVER_18_CONFIRMATION,
            aliases=["over 18", "18 years or older", "adult", "of legal age"],
            context_keywords=["18", "adult", "age", "years", "older"],
            typical_field_types=["checkbox"],
            is_attestational=True,  # Legal attestation
            max_expected_length=None
        ),
        SemanticLabel.UNDER_PENALTY_OF_PERJURY: LabelMetadata(
            label=SemanticLabel.UNDER_PENALTY_OF_PERJURY,
            aliases=["under penalty of perjury", "declare under penalty", "perjury"],
            context_keywords=["perjury", "penalty", "declare", "swear", "affirm"],
            typical_field_types=["checkbox", "line"],
            is_attestational=True,  # Legal attestation
            max_expected_length=None
        ),
        SemanticLabel.UNKNOWN_FIELD: LabelMetadata(
            label=SemanticLabel.UNKNOWN_FIELD,
            aliases=[],
            context_keywords=[],
            typical_field_types=["line", "checkbox", "table_cell"],
            is_attestational=False,
            max_expected_length=None
        ),
    }
    
    # Add remaining labels with basic metadata
    # (In production, all would have rich metadata like above)
    _BASIC_LABELS = [
        SemanticLabel.MIDDLE_NAME,
        SemanticLabel.PARTY_NAME,
        SemanticLabel.PLAINTIFF_NAME,
        SemanticLabel.DEFENDANT_NAME,
        SemanticLabel.PETITIONER_NAME,
        SemanticLabel.RESPONDENT_NAME,
        SemanticLabel.WITNESS_NAME,
        SemanticLabel.JUDGE_NAME,
        SemanticLabel.DOCKET_NUMBER,
        SemanticLabel.COURT_DEPARTMENT,
        SemanticLabel.COURT_DIVISION,
        SemanticLabel.COURT_ADDRESS,
        SemanticLabel.JUDICIAL_DISTRICT,
        SemanticLabel.COUNTY,
        SemanticLabel.DATE_OF_BIRTH,
        SemanticLabel.FILING_DATE,
        SemanticLabel.HEARING_DATE,
        SemanticLabel.HEARING_TIME,
        SemanticLabel.SERVICE_DATE,
        SemanticLabel.INCIDENT_DATE,
        SemanticLabel.STREET_ADDRESS,
        SemanticLabel.CITY,
        SemanticLabel.STATE,
        SemanticLabel.ZIP_CODE,
        SemanticLabel.FAX,
        SemanticLabel.INITIALS,
        SemanticLabel.DECLARATION_CONFIRMATION,
        SemanticLabel.NOTARY_SIGNATURE,
        SemanticLabel.NOTARY_COMMISSION_EXPIRY,
        SemanticLabel.AMOUNT,
        SemanticLabel.DAMAGES_AMOUNT,
        SemanticLabel.FILING_FEE,
        SemanticLabel.BOND_AMOUNT,
        SemanticLabel.DESCRIPTION,
        SemanticLabel.CAUSE_OF_ACTION,
        SemanticLabel.RELIEF_REQUESTED,
        SemanticLabel.FACTS,
        SemanticLabel.ADDITIONAL_INFORMATION,
        SemanticLabel.EXHIBIT_NUMBER,
        SemanticLabel.PAGE_NUMBER,
        SemanticLabel.ATTACHMENT_NUMBER,
        SemanticLabel.CHECKBOX_REPRESENTED_BY_ATTORNEY,
        SemanticLabel.CHECKBOX_PRO_SE,
        SemanticLabel.CHECKBOX_FEE_WAIVER,
        SemanticLabel.CHECKBOX_INTERPRETER_NEEDED,
        SemanticLabel.CHECKBOX_DISABILITY_ACCOMMODATION,
        SemanticLabel.CHECKBOX_CONSENT,
        SemanticLabel.CHECKBOX_CERTIFICATION,
    ]
    
    def __init__(self):
        """Initialize ontology with complete label set."""
        # Add basic metadata for labels not explicitly defined
        for label in self._BASIC_LABELS:
            if label not in self._LABEL_METADATA:
                self._LABEL_METADATA[label] = LabelMetadata(
                    label=label,
                    aliases=[label.value.lower().replace("_", " ")],
                    context_keywords=label.value.lower().split("_"),
                    typical_field_types=["checkbox"] if "CHECKBOX" in label.value else ["line"],
                    is_attestational="CHECKBOX" in label.value or label in [
                        SemanticLabel.INITIALS,
                        SemanticLabel.NOTARY_SIGNATURE,
                        SemanticLabel.DECLARATION_CONFIRMATION,
                    ],
                    max_expected_length=None
                )
        
        # Build reverse lookup for aliases
        self._alias_to_label: Dict[str, SemanticLabel] = {}
        for label, metadata in self._LABEL_METADATA.items():
            for alias in metadata.aliases:
                self._alias_to_label[alias.lower()] = label
    
    @property
    def all_labels(self) -> List[SemanticLabel]:
        """Return all valid semantic labels."""
        return list(SemanticLabel)
    
    @property
    def label_names(self) -> List[str]:
        """Return all label names as strings."""
        return [label.value for label in SemanticLabel]
    
    @property
    def attestational_labels(self) -> Set[SemanticLabel]:
        """Return labels that have legal attestation significance."""
        return {
            label for label, meta in self._LABEL_METADATA.items()
            if meta.is_attestational
        }
    
    def get_metadata(self, label: SemanticLabel) -> Optional[LabelMetadata]:
        """Get metadata for a label."""
        return self._LABEL_METADATA.get(label)
    
    def lookup_by_alias(self, text: str) -> Optional[SemanticLabel]:
        """
        Look up a label by alias text.
        
        Args:
            text: The text to look up (case-insensitive)
        
        Returns:
            The matching SemanticLabel or None
        """
        return self._alias_to_label.get(text.lower().strip())
    
    def is_valid_label(self, label_str: str) -> bool:
        """Check if a string is a valid label in the ontology."""
        try:
            SemanticLabel(label_str)
            return True
        except ValueError:
            return False
    
    def get_label_id(self, label: SemanticLabel) -> int:
        """
        Get numeric ID for a label (for LayoutLM classification).
        
        Returns:
            Integer ID corresponding to the label's position in the enum.
        """
        return list(SemanticLabel).index(label)
    
    def get_label_from_id(self, label_id: int) -> SemanticLabel:
        """
        Get label from numeric ID.
        
        Args:
            label_id: Integer ID
        
        Returns:
            The corresponding SemanticLabel
        """
        labels = list(SemanticLabel)
        if 0 <= label_id < len(labels):
            return labels[label_id]
        return SemanticLabel.UNKNOWN_FIELD
    
    @property
    def num_labels(self) -> int:
        """Return total number of labels in ontology."""
        return len(SemanticLabel)


# Global singleton instance
ONTOLOGY = LegalFieldOntology()

