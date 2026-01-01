"""BioBERT-based NER model for medical entity recognition."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class BioBERTNERModel(nn.Module):
    """BioBERT-based Named Entity Recognition model."""
    
    def __init__(self, config: Dict):
        """Initialize BioBERT NER model.
        
        Args:
            config: Model configuration dictionary.
        """
        super().__init__()
        
        self.config = config
        self.model_name = config.get('model_name', 'dmis-lab/biobert-base-cased-v1.1')
        self.max_length = config.get('max_length', 512)
        self.num_labels = config.get('num_labels', 9)  # BILOU tags
        self.dropout = config.get('dropout', 0.1)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.bert = AutoModel.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Classification head
        self.dropout_layer = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized BioBERT NER model with {self.num_labels} labels")
    
    def _init_weights(self) -> None:
        """Initialize classifier weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            input_ids: Tokenized input text.
            attention_mask: Attention mask for input.
            labels: Ground truth labels (optional).
            
        Returns:
            Dictionary containing logits and optionally loss.
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get sequence output
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout
        sequence_output = self.dropout_layer(sequence_output)
        
        # Get logits
        logits = self.classifier(sequence_output)
        
        outputs_dict = {'logits': logits}
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            outputs_dict['loss'] = loss
        
        return outputs_dict
    
    def predict(self, texts: List[str]) -> List[List[Dict]]:
        """Predict entities in texts.
        
        Args:
            texts: List of input texts.
            
        Returns:
            List of predicted entities for each text.
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                device = next(self.parameters()).device
                input_ids = encoding['input_ids'].to(device)
                attention_mask = encoding['attention_mask'].to(device)
                
                # Forward pass
                outputs = self.forward(input_ids, attention_mask)
                logits = outputs['logits']
                
                # Get predictions
                predicted_labels = torch.argmax(logits, dim=-1)
                
                # Convert to entities
                entities = self._convert_predictions_to_entities(
                    text, encoding, predicted_labels[0]
                )
                
                predictions.append(entities)
        
        return predictions
    
    def _convert_predictions_to_entities(
        self,
        text: str,
        encoding: Dict,
        predictions: torch.Tensor
    ) -> List[Dict]:
        """Convert model predictions to entity format.
        
        Args:
            text: Original text.
            encoding: Tokenizer encoding.
            predictions: Predicted labels.
            
        Returns:
            List of predicted entities.
        """
        entities = []
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
        
        current_entity = None
        
        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            
            pred_label = self._id_to_label(pred.item())
            
            if pred_label.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = pred_label[2:]
                current_entity = {
                    'text': token,
                    'label': entity_type,
                    'start': encoding.token_to_chars(i)[0] if hasattr(encoding, 'token_to_chars') else 0,
                    'end': encoding.token_to_chars(i)[1] if hasattr(encoding, 'token_to_chars') else len(token),
                    'confidence': torch.softmax(predictions[i], dim=0)[pred].item()
                }
            
            elif pred_label.startswith('I-') and current_entity:
                # Continue current entity
                current_entity['text'] += token.replace('##', '')
                current_entity['end'] = encoding.token_to_chars(i)[1] if hasattr(encoding, 'token_to_chars') else current_entity['end'] + len(token)
            
            elif pred_label == 'O':
                # End current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add final entity if exists
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def _id_to_label(self, label_id: int) -> str:
        """Convert label ID to label string.
        
        Args:
            label_id: Label ID.
            
        Returns:
            Label string.
        """
        # BILOU tagging scheme
        label_map = {
            0: 'O',
            1: 'B-DISEASE',
            2: 'I-DISEASE',
            3: 'B-CHEMICAL',
            4: 'I-CHEMICAL',
            5: 'B-SYMPTOM',
            6: 'I-SYMPTOM',
            7: 'B-PROCEDURE',
            8: 'I-PROCEDURE'
        }
        return label_map.get(label_id, 'O')
    
    def get_attention_weights(self, text: str) -> torch.Tensor:
        """Get attention weights for explainability.
        
        Args:
            text: Input text.
            
        Returns:
            Attention weights tensor.
        """
        self.eval()
        
        with torch.no_grad():
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            device = next(self.parameters()).device
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True
            )
            
            # Get attention weights from last layer
            attention_weights = outputs.attentions[-1]  # [batch_size, num_heads, seq_len, seq_len]
            
            return attention_weights[0]  # Remove batch dimension
