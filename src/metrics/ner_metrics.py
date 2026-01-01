"""Evaluation metrics for medical entity recognition."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging

logger = logging.getLogger(__name__)


class NERMetrics:
    """Metrics calculator for Named Entity Recognition."""
    
    def __init__(self, entity_types: List[str]):
        """Initialize metrics calculator.
        
        Args:
            entity_types: List of entity types to evaluate.
        """
        self.entity_types = entity_types
        self.label_to_id = self._create_label_mapping()
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
    
    def _create_label_mapping(self) -> Dict[str, int]:
        """Create label to ID mapping using BILOU scheme.
        
        Returns:
            Dictionary mapping labels to IDs.
        """
        label_map = {'O': 0}
        
        for entity_type in self.entity_types:
            label_map[f'B-{entity_type}'] = len(label_map)
            label_map[f'I-{entity_type}'] = len(label_map)
        
        return label_map
    
    def align_predictions(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align predictions with labels, ignoring padding tokens.
        
        Args:
            predictions: Model predictions.
            labels: Ground truth labels.
            attention_mask: Attention mask.
            
        Returns:
            Tuple of (aligned_predictions, aligned_labels).
        """
        # Flatten tensors
        flat_predictions = predictions.view(-1)
        flat_labels = labels.view(-1)
        flat_mask = attention_mask.view(-1)
        
        # Filter out padding tokens
        valid_indices = flat_mask == 1
        aligned_predictions = flat_predictions[valid_indices].cpu().numpy()
        aligned_labels = flat_labels[valid_indices].cpu().numpy()
        
        return aligned_predictions, aligned_labels
    
    def calculate_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate comprehensive NER metrics.
        
        Args:
            predictions: Model predictions.
            labels: Ground truth labels.
            attention_mask: Attention mask.
            
        Returns:
            Dictionary of calculated metrics.
        """
        # Align predictions and labels
        aligned_preds, aligned_labels = self.align_predictions(
            predictions, labels, attention_mask
        )
        
        # Overall metrics
        accuracy = accuracy_score(aligned_labels, aligned_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            aligned_labels, aligned_preds, average='weighted', zero_division=0
        )
        
        # Per-entity metrics
        entity_metrics = {}
        for entity_type in self.entity_types:
            entity_precision, entity_recall, entity_f1, _ = precision_recall_fscore_support(
                aligned_labels, aligned_preds, 
                labels=[self.label_to_id[f'B-{entity_type}'], self.label_to_id[f'I-{entity_type}']],
                average='macro', zero_division=0
            )
            
            entity_metrics[f'{entity_type}_precision'] = entity_precision
            entity_metrics[f'{entity_type}_recall'] = entity_recall
            entity_metrics[f'{entity_type}_f1'] = entity_f1
        
        # Micro-averaged metrics
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            aligned_labels, aligned_preds, average='micro', zero_division=0
        )
        
        # Macro-averaged metrics
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            aligned_labels, aligned_preds, average='macro', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            **entity_metrics
        }
        
        return metrics
    
    def calculate_entity_level_metrics(
        self,
        true_entities: List[List[Dict]],
        pred_entities: List[List[Dict]]
    ) -> Dict[str, float]:
        """Calculate entity-level metrics (exact match).
        
        Args:
            true_entities: List of true entities for each text.
            pred_entities: List of predicted entities for each text.
            
        Returns:
            Dictionary of entity-level metrics.
        """
        total_true = 0
        total_pred = 0
        total_correct = 0
        
        entity_type_stats = {entity_type: {'true': 0, 'pred': 0, 'correct': 0} 
                           for entity_type in self.entity_types}
        
        for true_ents, pred_ents in zip(true_entities, pred_entities):
            # Convert to sets for comparison
            true_set = set()
            pred_set = set()
            
            for ent in true_ents:
                entity_key = (ent['text'].lower(), ent['label'])
                true_set.add(entity_key)
                entity_type_stats[ent['label']]['true'] += 1
            
            for ent in pred_ents:
                entity_key = (ent['text'].lower(), ent['label'])
                pred_set.add(entity_key)
                entity_type_stats[ent['label']]['pred'] += 1
            
            # Count matches
            matches = true_set.intersection(pred_set)
            total_correct += len(matches)
            
            # Count per-entity-type matches
            for match in matches:
                entity_type = match[1]
                entity_type_stats[entity_type]['correct'] += 1
            
            total_true += len(true_set)
            total_pred += len(pred_set)
        
        # Calculate overall metrics
        precision = total_correct / total_pred if total_pred > 0 else 0
        recall = total_correct / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate per-entity-type metrics
        entity_metrics = {}
        for entity_type, stats in entity_type_stats.items():
            entity_precision = stats['correct'] / stats['pred'] if stats['pred'] > 0 else 0
            entity_recall = stats['correct'] / stats['true'] if stats['true'] > 0 else 0
            entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0
            
            entity_metrics[f'{entity_type}_entity_precision'] = entity_precision
            entity_metrics[f'{entity_type}_entity_recall'] = entity_recall
            entity_metrics[f'{entity_type}_entity_f1'] = entity_f1
        
        metrics = {
            'entity_precision': precision,
            'entity_recall': recall,
            'entity_f1': f1,
            **entity_metrics
        }
        
        return metrics
    
    def calculate_calibration_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate calibration metrics.
        
        Args:
            predictions: Model predictions (logits).
            labels: Ground truth labels.
            attention_mask: Attention mask.
            
        Returns:
            Dictionary of calibration metrics.
        """
        # Get probabilities
        probs = torch.softmax(predictions, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        
        # Align with labels
        aligned_probs, aligned_labels = self.align_predictions(
            max_probs, labels, attention_mask
        )
        
        # Calculate Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (aligned_probs > bin_lower) & (aligned_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (aligned_labels[in_bin] == np.argmax(predictions.view(-1, predictions.size(-1))[in_bin], axis=1)).mean()
                avg_confidence_in_bin = aligned_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Calculate Brier Score
        correct_predictions = (aligned_labels == np.argmax(predictions.view(-1, predictions.size(-1)), axis=1)).astype(float)
        brier_score = np.mean((aligned_probs - correct_predictions) ** 2)
        
        return {
            'ece': ece,
            'brier_score': brier_score,
            'avg_confidence': np.mean(aligned_probs),
            'accuracy': np.mean(correct_predictions)
        }
