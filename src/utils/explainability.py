"""Explainability utilities for medical entity recognition."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """Visualize attention weights for model explainability."""
    
    def __init__(self, model, tokenizer):
        """Initialize attention visualizer.
        
        Args:
            model: Trained NER model.
            tokenizer: Tokenizer used by the model.
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def get_attention_weights(self, text: str) -> torch.Tensor:
        """Get attention weights for input text.
        
        Args:
            text: Input text.
            
        Returns:
            Attention weights tensor.
        """
        self.model.eval()
        
        with torch.no_grad():
            # Tokenize input
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            device = next(self.model.parameters()).device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Get attention weights
            attention_weights = self.model.get_attention_weights(text)
            
            return attention_weights
    
    def visualize_attention(
        self,
        text: str,
        attention_weights: torch.Tensor,
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize attention weights as heatmap.
        
        Args:
            text: Input text.
            attention_weights: Attention weights tensor.
            layer_idx: Which layer to visualize.
            head_idx: Which attention head to visualize (None for average).
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        
        # Get attention weights for specified layer
        if layer_idx < 0:
            layer_idx = attention_weights.shape[0] + layer_idx
        
        layer_attention = attention_weights[layer_idx]  # [num_heads, seq_len, seq_len]
        
        # Average across heads or select specific head
        if head_idx is None:
            attention_matrix = layer_attention.mean(dim=0)  # [seq_len, seq_len]
        else:
            attention_matrix = layer_attention[head_idx]  # [seq_len, seq_len]
        
        # Convert to numpy and remove special tokens
        attention_matrix = attention_matrix.cpu().numpy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            attention_matrix,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Attention Weight'}
        )
        
        ax.set_title(f'Attention Visualization - Layer {layer_idx}')
        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Attention visualization saved to {save_path}")
        
        return fig


class EntityAttributionAnalyzer:
    """Analyze entity attribution and importance."""
    
    def __init__(self, model, tokenizer):
        """Initialize entity attribution analyzer.
        
        Args:
            model: Trained NER model.
            tokenizer: Tokenizer used by the model.
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def get_entity_importance_scores(
        self,
        text: str,
        entities: List[Dict]
    ) -> List[Dict]:
        """Calculate importance scores for detected entities.
        
        Args:
            text: Input text.
            entities: List of detected entities.
            
        Returns:
            List of entities with importance scores.
        """
        self.model.eval()
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        device = next(self.model.parameters()).device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            probabilities = torch.softmax(logits, dim=-1)
        
        # Calculate importance scores for each entity
        entities_with_scores = []
        
        for entity in entities:
            # Find tokens corresponding to entity
            entity_tokens = self.tokenizer.tokenize(entity['text'])
            entity_start_token = None
            entity_end_token = None
            
            # Find token positions
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            for i, token in enumerate(tokens):
                if token.lower() == entity_tokens[0].lower():
                    entity_start_token = i
                    break
            
            if entity_start_token is not None:
                # Calculate average probability for entity tokens
                entity_probs = []
                for i in range(len(entity_tokens)):
                    if entity_start_token + i < len(tokens):
                        token_probs = probabilities[0, entity_start_token + i]
                        entity_probs.append(token_probs.max().item())
                
                # Calculate importance score
                avg_confidence = np.mean(entity_probs) if entity_probs else 0.0
                max_confidence = np.max(entity_probs) if entity_probs else 0.0
                
                entity_with_score = entity.copy()
                entity_with_score.update({
                    'avg_confidence': avg_confidence,
                    'max_confidence': max_confidence,
                    'importance_score': avg_confidence * max_confidence
                })
                
                entities_with_scores.append(entity_with_score)
        
        return entities_with_scores
    
    def visualize_entity_importance(
        self,
        text: str,
        entities_with_scores: List[Dict],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Visualize entity importance scores.
        
        Args:
            text: Input text.
            entities_with_scores: Entities with importance scores.
            save_path: Path to save the plot.
            
        Returns:
            Matplotlib figure.
        """
        if not entities_with_scores:
            logger.warning("No entities to visualize")
            return None
        
        # Extract data for plotting
        entity_texts = [e['text'] for e in entities_with_scores]
        entity_labels = [e['label'] for e in entities_with_scores]
        importance_scores = [e['importance_score'] for e in entities_with_scores]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Importance scores by entity
        colors = plt.cm.Set3(np.linspace(0, 1, len(set(entity_labels))))
        label_colors = {label: colors[i] for i, label in enumerate(set(entity_labels))}
        
        bars = ax1.bar(range(len(entity_texts)), importance_scores, 
                      color=[label_colors[label] for label in entity_labels])
        ax1.set_xlabel('Entities')
        ax1.set_ylabel('Importance Score')
        ax1.set_title('Entity Importance Scores')
        ax1.set_xticks(range(len(entity_texts)))
        ax1.set_xticklabels(entity_texts, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, score in zip(bars, importance_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Plot 2: Importance scores by entity type
        entity_types = list(set(entity_labels))
        type_scores = []
        for entity_type in entity_types:
            type_entities = [e for e in entities_with_scores if e['label'] == entity_type]
            avg_score = np.mean([e['importance_score'] for e in type_entities])
            type_scores.append(avg_score)
        
        bars2 = ax2.bar(entity_types, type_scores, color=[label_colors[t] for t in entity_types])
        ax2.set_xlabel('Entity Type')
        ax2.set_ylabel('Average Importance Score')
        ax2.set_title('Average Importance by Entity Type')
        
        # Add value labels on bars
        for bar, score in zip(bars2, type_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Entity importance visualization saved to {save_path}")
        
        return fig


class ModelExplainer:
    """Main explainability interface."""
    
    def __init__(self, model, tokenizer):
        """Initialize model explainer.
        
        Args:
            model: Trained NER model.
            tokenizer: Tokenizer used by the model.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.attention_viz = AttentionVisualizer(model, tokenizer)
        self.attribution_analyzer = EntityAttributionAnalyzer(model, tokenizer)
    
    def explain_prediction(
        self,
        text: str,
        entities: List[Dict],
        output_dir: str = "assets"
    ) -> Dict[str, str]:
        """Generate comprehensive explanation for model prediction.
        
        Args:
            text: Input text.
            entities: Detected entities.
            output_dir: Directory to save visualizations.
            
        Returns:
            Dictionary with paths to saved visualizations.
        """
        from pathlib import Path
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        try:
            # Get attention weights
            attention_weights = self.attention_viz.get_attention_weights(text)
            
            # Visualize attention
            attention_fig = self.attention_viz.visualize_attention(
                text, attention_weights, save_path=f"{output_dir}/attention_heatmap.png"
            )
            saved_paths['attention'] = f"{output_dir}/attention_heatmap.png"
            
            # Analyze entity importance
            entities_with_scores = self.attribution_analyzer.get_entity_importance_scores(
                text, entities
            )
            
            # Visualize entity importance
            importance_fig = self.attribution_analyzer.visualize_entity_importance(
                text, entities_with_scores, save_path=f"{output_dir}/entity_importance.png"
            )
            saved_paths['entity_importance'] = f"{output_dir}/entity_importance.png"
            
            logger.info(f"Generated explanations for {len(entities)} entities")
            
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
        
        return saved_paths
