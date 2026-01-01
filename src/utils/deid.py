"""De-identification utilities for protecting PHI/PII in clinical text."""

import re
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DeIdentifier:
    """De-identification utility for clinical text."""
    
    def __init__(self, enable_deid: bool = True):
        """Initialize de-identifier.
        
        Args:
            enable_deid: Whether to enable de-identification.
        """
        self.enable_deid = enable_deid
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for PHI detection."""
        # Common PHI patterns
        self.patterns = {
            'ssn': re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b'),
            'phone': re.compile(r'\b\d{3}-?\d{3}-?\d{4}\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'date': re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'),
            'mrn': re.compile(r'\bMRN\s*:?\s*\d+\b', re.IGNORECASE),
            'patient_id': re.compile(r'\bPatient\s+ID\s*:?\s*\d+\b', re.IGNORECASE),
            'age': re.compile(r'\b(\d{1,3})\s*years?\s*old\b', re.IGNORECASE),
            'zipcode': re.compile(r'\b\d{5}(?:-\d{4})?\b'),
        }
        
        # Replacement mappings
        self.replacements = {
            'ssn': '[SSN]',
            'phone': '[PHONE]',
            'email': '[EMAIL]',
            'date': '[DATE]',
            'mrn': '[MRN]',
            'patient_id': '[PATIENT_ID]',
            'age': '[AGE]',
            'zipcode': '[ZIPCODE]',
        }
    
    def deidentify_text(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """De-identify text by replacing PHI with placeholders.
        
        Args:
            text: Input clinical text.
            
        Returns:
            Tuple of (de-identified text, list of replaced entities).
        """
        if not self.enable_deid:
            return text, []
        
        deidentified_text = text
        replaced_entities = []
        
        for pattern_name, pattern in self.patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                original = match.group()
                replacement = self.replacements[pattern_name]
                deidentified_text = deidentified_text.replace(original, replacement)
                replaced_entities.append({
                    'type': pattern_name,
                    'original': original,
                    'replacement': replacement,
                    'start': match.start(),
                    'end': match.end()
                })
        
        logger.debug(f"De-identified {len(replaced_entities)} PHI entities")
        return deidentified_text, replaced_entities
    
    def redact_entities(self, text: str, entities: List[Dict]) -> str:
        """Redact specific entities from text.
        
        Args:
            text: Input text.
            entities: List of entity dictionaries with 'start' and 'end' keys.
            
        Returns:
            Text with entities redacted.
        """
        if not self.enable_deid:
            return text
        
        # Sort entities by start position (descending) to avoid index shifts
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        redacted_text = text
        for entity in sorted_entities:
            start = entity['start']
            end = entity['end']
            redacted_text = redacted_text[:start] + '[REDACTED]' + redacted_text[end:]
        
        return redacted_text


def validate_no_phi(text: str) -> bool:
    """Validate that text contains no PHI.
    
    Args:
        text: Text to validate.
        
    Returns:
        True if no PHI detected, False otherwise.
    """
    deid = DeIdentifier(enable_deid=True)
    _, replaced_entities = deid.deidentify_text(text)
    return len(replaced_entities) == 0


def sanitize_log_message(message: str) -> str:
    """Sanitize log message to remove potential PHI.
    
    Args:
        message: Log message to sanitize.
        
    Returns:
        Sanitized log message.
    """
    deid = DeIdentifier(enable_deid=True)
    sanitized, _ = deid.deidentify_text(message)
    return sanitized
