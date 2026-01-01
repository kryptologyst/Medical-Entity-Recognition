"""Unit tests for medical entity recognition models."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.models.biobert_model import BioBERTNERModel
from src.data.synthetic_dataset import SyntheticNERDataset, ClinicalNote, Entity
from src.utils.deid import DeIdentifier
from src.metrics.ner_metrics import NERMetrics


class TestBioBERTNERModel:
    """Test cases for BioBERT NER model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        config = {
            'model_name': 'dmis-lab/biobert-base-cased-v1.1',
            'max_length': 512,
            'num_labels': 9,
            'dropout': 0.1
        }
        
        with patch('src.models.biobert_model.AutoTokenizer') as mock_tokenizer, \
             patch('src.models.biobert_model.AutoModel') as mock_model:
            
            mock_tokenizer.from_pretrained.return_value = Mock()
            mock_model.from_pretrained.return_value = Mock()
            
            model = BioBERTNERModel(config)
            
            assert model.model_name == config['model_name']
            assert model.max_length == config['max_length']
            assert model.num_labels == config['num_labels']
            assert model.dropout == config['dropout']
    
    def test_forward_pass(self):
        """Test model forward pass."""
        config = {
            'model_name': 'dmis-lab/biobert-base-cased-v1.1',
            'max_length': 512,
            'num_labels': 9,
            'dropout': 0.1
        }
        
        with patch('src.models.biobert_model.AutoTokenizer') as mock_tokenizer, \
             patch('src.models.biobert_model.AutoModel') as mock_model:
            
            # Mock tokenizer
            mock_tokenizer_instance = Mock()
            mock_tokenizer_instance.pad_token = None
            mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
            
            # Mock BERT model
            mock_bert_instance = Mock()
            mock_bert_instance.config.hidden_size = 768
            mock_bert_instance.return_value = Mock(last_hidden_state=torch.randn(1, 10, 768))
            mock_model.from_pretrained.return_value = mock_bert_instance
            
            model = BioBERTNERModel(config)
            
            # Test forward pass
            input_ids = torch.randint(0, 1000, (1, 10))
            attention_mask = torch.ones(1, 10)
            labels = torch.randint(0, 9, (1, 10))
            
            outputs = model(input_ids, attention_mask, labels)
            
            assert 'logits' in outputs
            assert 'loss' in outputs
            assert outputs['logits'].shape == (1, 10, 9)
            assert isinstance(outputs['loss'], torch.Tensor)


class TestSyntheticNERDataset:
    """Test cases for synthetic dataset generation."""
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        config = {
            'entity_templates': {
                'disease': ['hypertension', 'diabetes'],
                'chemical': ['metformin', 'insulin']
            },
            'num_samples': 100,
            'min_length': 50,
            'max_length': 500
        }
        
        dataset = SyntheticNERDataset(config)
        
        assert dataset.num_samples == 100
        assert dataset.min_length == 50
        assert dataset.max_length == 500
        assert 'disease' in dataset.entity_templates
        assert 'chemical' in dataset.entity_templates
    
    def test_entity_generation(self):
        """Test entity generation."""
        config = {
            'entity_templates': {
                'disease': ['hypertension', 'diabetes'],
                'chemical': ['metformin', 'insulin']
            }
        }
        
        dataset = SyntheticNERDataset(config)
        
        # Test disease entity generation
        disease_entity = dataset.generate_entity('disease')
        assert disease_entity is not None
        assert disease_entity.label == 'disease'
        assert disease_entity.text in ['hypertension', 'diabetes']
        
        # Test chemical entity generation
        chemical_entity = dataset.generate_entity('chemical')
        assert chemical_entity is not None
        assert chemical_entity.label == 'chemical'
        assert chemical_entity.text in ['metformin', 'insulin']
    
    def test_clinical_note_generation(self):
        """Test clinical note generation."""
        config = {
            'entity_templates': {
                'disease': ['hypertension'],
                'chemical': ['metformin'],
                'symptom': ['fatigue'],
                'procedure': ['blood test']
            }
        }
        
        dataset = SyntheticNERDataset(config)
        
        note = dataset.generate_clinical_note("test_note_001")
        
        assert isinstance(note, ClinicalNote)
        assert note.note_id == "test_note_001"
        assert len(note.text) > 0
        assert isinstance(note.entities, list)
    
    def test_dataset_generation(self):
        """Test full dataset generation."""
        config = {
            'entity_templates': {
                'disease': ['hypertension'],
                'chemical': ['metformin']
            },
            'num_samples': 5
        }
        
        dataset = SyntheticNERDataset(config)
        notes = dataset.generate_dataset()
        
        assert len(notes) == 5
        assert all(isinstance(note, ClinicalNote) for note in notes)
        assert all(len(note.text) > 0 for note in notes)


class TestDeIdentifier:
    """Test cases for de-identification utilities."""
    
    def test_deidentifier_initialization(self):
        """Test de-identifier initialization."""
        deid = DeIdentifier(enable_deid=True)
        assert deid.enable_deid is True
        
        deid_disabled = DeIdentifier(enable_deid=False)
        assert deid_disabled.enable_deid is False
    
    def test_phi_detection(self):
        """Test PHI detection."""
        deid = DeIdentifier(enable_deid=True)
        
        # Test SSN detection
        text_with_ssn = "Patient SSN: 123-45-6789"
        deidentified_text, phi_entities = deid.deidentify_text(text_with_ssn)
        
        assert "[SSN]" in deidentified_text
        assert len(phi_entities) > 0
        assert any(entity['type'] == 'ssn' for entity in phi_entities)
    
    def test_phone_detection(self):
        """Test phone number detection."""
        deid = DeIdentifier(enable_deid=True)
        
        text_with_phone = "Contact: 555-123-4567"
        deidentified_text, phi_entities = deid.deidentify_text(text_with_phone)
        
        assert "[PHONE]" in deidentified_text
        assert any(entity['type'] == 'phone' for entity in phi_entities)
    
    def test_email_detection(self):
        """Test email detection."""
        deid = DeIdentifier(enable_deid=True)
        
        text_with_email = "Email: patient@hospital.com"
        deidentified_text, phi_entities = deid.deidentify_text(text_with_email)
        
        assert "[EMAIL]" in deidentified_text
        assert any(entity['type'] == 'email' for entity in phi_entities)
    
    def test_no_phi_text(self):
        """Test text without PHI."""
        deid = DeIdentifier(enable_deid=True)
        
        clean_text = "Patient diagnosed with hypertension."
        deidentified_text, phi_entities = deid.deidentify_text(clean_text)
        
        assert deidentified_text == clean_text
        assert len(phi_entities) == 0
    
    def test_deid_disabled(self):
        """Test de-identification when disabled."""
        deid = DeIdentifier(enable_deid=False)
        
        text_with_phi = "Patient SSN: 123-45-6789"
        deidentified_text, phi_entities = deid.deidentify_text(text_with_phi)
        
        assert deidentified_text == text_with_phi
        assert len(phi_entities) == 0


class TestNERMetrics:
    """Test cases for NER evaluation metrics."""
    
    def test_metrics_initialization(self):
        """Test metrics calculator initialization."""
        entity_types = ['DISEASE', 'CHEMICAL', 'SYMPTOM']
        metrics = NERMetrics(entity_types)
        
        assert metrics.entity_types == entity_types
        assert 'O' in metrics.label_to_id
        assert 'B-DISEASE' in metrics.label_to_id
        assert 'I-DISEASE' in metrics.label_to_id
    
    def test_label_mapping(self):
        """Test label to ID mapping."""
        entity_types = ['DISEASE', 'CHEMICAL']
        metrics = NERMetrics(entity_types)
        
        # Check BILOU mapping
        assert metrics.label_to_id['O'] == 0
        assert metrics.label_to_id['B-DISEASE'] == 1
        assert metrics.label_to_id['I-DISEASE'] == 2
        assert metrics.label_to_id['B-CHEMICAL'] == 3
        assert metrics.label_to_id['I-CHEMICAL'] == 4
    
    def test_align_predictions(self):
        """Test prediction alignment."""
        entity_types = ['DISEASE']
        metrics = NERMetrics(entity_types)
        
        # Create mock data
        predictions = torch.randn(2, 5, 5)  # [batch_size, seq_len, num_labels]
        labels = torch.randint(0, 5, (2, 5))
        attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
        
        aligned_preds, aligned_labels = metrics.align_predictions(
            predictions, labels, attention_mask
        )
        
        # Should have 5 valid tokens (3 + 2, excluding padding)
        assert len(aligned_preds) == 5
        assert len(aligned_labels) == 5
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        entity_types = ['DISEASE']
        metrics = NERMetrics(entity_types)
        
        # Create mock data
        predictions = torch.randn(1, 3, 5)  # [batch_size, seq_len, num_labels]
        labels = torch.tensor([[0, 1, 2]])  # O, B-DISEASE, I-DISEASE
        attention_mask = torch.ones(1, 3)
        
        calculated_metrics = metrics.calculate_metrics(
            predictions, labels, attention_mask
        )
        
        # Check that all expected metrics are present
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1']
        for metric in expected_metrics:
            assert metric in calculated_metrics
            assert isinstance(calculated_metrics[metric], float)


if __name__ == '__main__':
    pytest.main([__file__])
