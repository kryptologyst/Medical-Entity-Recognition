"""Synthetic dataset generation for medical entity recognition."""

import random
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents a medical entity."""
    text: str
    label: str
    start: int
    end: int


@dataclass
class ClinicalNote:
    """Represents a clinical note with entities."""
    text: str
    entities: List[Entity]
    note_id: str


class SyntheticNERDataset:
    """Generates synthetic clinical notes for NER training."""
    
    def __init__(self, config: Dict):
        """Initialize dataset generator.
        
        Args:
            config: Dataset configuration.
        """
        self.config = config
        self.entity_templates = config.get('entity_templates', {})
        self.num_samples = config.get('num_samples', 1000)
        self.min_length = config.get('min_length', 50)
        self.max_length = config.get('max_length', 500)
        self.num_entities_per_text = config.get('num_entities_per_text', 3)
        
        # Clinical note templates
        self.note_templates = [
            "Patient presents with {symptoms}. Physical examination reveals {findings}. Treatment plan includes {treatment}.",
            "The patient was diagnosed with {disease}. Current medications include {chemical}. Patient reports {symptoms}.",
            "Follow-up visit for {disease}. Patient is responding well to {chemical}. No new {symptoms} reported.",
            "Emergency department visit for {symptoms}. Diagnostic workup included {procedure}. Patient was treated with {chemical}.",
            "Patient history of {disease}. Recent {procedure} showed {findings}. Current therapy with {chemical}.",
        ]
        
        # Clinical findings templates
        self.findings_templates = [
            "normal vital signs",
            "elevated blood pressure",
            "irregular heartbeat",
            "clear lung sounds",
            "normal heart sounds",
            "mild tenderness",
            "no acute distress",
            "alert and oriented",
        ]
    
    def generate_entity(self, entity_type: str) -> Entity:
        """Generate a random entity of given type.
        
        Args:
            entity_type: Type of entity to generate.
            
        Returns:
            Generated entity.
        """
        templates = self.entity_templates.get(entity_type, [])
        if not templates:
            return None
        
        text = random.choice(templates)
        return Entity(
            text=text,
            label=entity_type,
            start=0,  # Will be set when inserted into text
            end=len(text)
        )
    
    def generate_clinical_note(self, note_id: str) -> ClinicalNote:
        """Generate a synthetic clinical note.
        
        Args:
            note_id: Unique identifier for the note.
            
        Returns:
            Generated clinical note.
        """
        # Choose a random template
        template = random.choice(self.note_templates)
        
        # Generate entities for each placeholder
        entities = []
        text_parts = []
        current_pos = 0
        
        # Split template by placeholders
        parts = template.split('{')
        for i, part in enumerate(parts):
            if '}' in part:
                placeholder, rest = part.split('}', 1)
                placeholder = placeholder.strip()
                
                # Generate entity based on placeholder
                if placeholder in ['symptoms']:
                    entity = self.generate_entity('symptom')
                elif placeholder in ['disease']:
                    entity = self.generate_entity('disease')
                elif placeholder in ['chemical', 'treatment']:
                    entity = self.generate_entity('chemical')
                elif placeholder in ['procedure']:
                    entity = self.generate_entity('procedure')
                elif placeholder in ['findings']:
                    entity_text = random.choice(self.findings_templates)
                    entity = Entity(
                        text=entity_text,
                        label='FINDING',
                        start=current_pos,
                        end=current_pos + len(entity_text)
                    )
                else:
                    entity = None
                
                if entity:
                    entity.start = current_pos
                    entity.end = current_pos + len(entity.text)
                    entities.append(entity)
                    text_parts.append(entity.text)
                    current_pos += len(entity.text)
                else:
                    text_parts.append(f"[{placeholder}]")
                    current_pos += len(f"[{placeholder}]")
                
                # Add rest of the part
                if rest:
                    text_parts.append(rest)
                    current_pos += len(rest)
            else:
                text_parts.append(part)
                current_pos += len(part)
        
        # Join all parts
        text = ''.join(text_parts)
        
        # Ensure minimum length
        if len(text) < self.min_length:
            additional_text = " Additional clinical information and observations were noted during the examination."
            text += additional_text
        
        # Truncate if too long
        if len(text) > self.max_length:
            text = text[:self.max_length]
            # Remove entities that are now out of bounds
            entities = [e for e in entities if e.end <= self.max_length]
        
        return ClinicalNote(
            text=text,
            entities=entities,
            note_id=note_id
        )
    
    def generate_dataset(self) -> List[ClinicalNote]:
        """Generate the complete dataset.
        
        Returns:
            List of generated clinical notes.
        """
        logger.info(f"Generating {self.num_samples} synthetic clinical notes")
        
        notes = []
        for i in range(self.num_samples):
            note_id = f"synthetic_note_{i:04d}"
            note = self.generate_clinical_note(note_id)
            notes.append(note)
        
        logger.info(f"Generated {len(notes)} clinical notes")
        return notes
    
    def save_dataset(self, notes: List[ClinicalNote], output_path: str) -> None:
        """Save dataset to file.
        
        Args:
            notes: List of clinical notes.
            output_path: Path to save the dataset.
        """
        data = []
        for note in notes:
            data.append({
                'note_id': note.note_id,
                'text': note.text,
                'entities': [
                    {
                        'text': entity.text,
                        'label': entity.label,
                        'start': entity.start,
                        'end': entity.end
                    }
                    for entity in note.entities
                ]
            })
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved dataset to {output_path}")
    
    def load_dataset(self, input_path: str) -> List[ClinicalNote]:
        """Load dataset from file.
        
        Args:
            input_path: Path to load the dataset from.
            
        Returns:
            List of clinical notes.
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        notes = []
        for item in data:
            entities = [
                Entity(
                    text=entity['text'],
                    label=entity['label'],
                    start=entity['start'],
                    end=entity['end']
                )
                for entity in item['entities']
            ]
            
            note = ClinicalNote(
                text=item['text'],
                entities=entities,
                note_id=item['note_id']
            )
            notes.append(note)
        
        logger.info(f"Loaded {len(notes)} clinical notes from {input_path}")
        return notes
