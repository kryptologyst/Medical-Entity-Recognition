"""Training pipeline for medical entity recognition."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.models.biobert_model import BioBERTNERModel
from src.metrics.ner_metrics import NERMetrics
from src.utils.deid import sanitize_log_message

logger = logging.getLogger(__name__)


class NERTrainer:
    """Trainer for Named Entity Recognition models."""
    
    def __init__(
        self,
        model: BioBERTNERModel,
        config: Dict,
        device: torch.device
    ):
        """Initialize trainer.
        
        Args:
            model: NER model to train.
            config: Training configuration.
            device: Device to train on.
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Initialize metrics
        self.metrics = NERMetrics(config.get('entity_types', ['DISEASE', 'CHEMICAL', 'SYMPTOM', 'PROCEDURE']))
        
        # Training state
        self.current_epoch = 0
        self.best_f1 = 0.0
        self.train_losses = []
        self.val_metrics = []
        
        logger.info(f"Initialized NER trainer on device {device}")
    
    def setup_optimizer_and_scheduler(
        self,
        train_dataloader: DataLoader,
        num_epochs: int
    ) -> Tuple[torch.optim.Optimizer, Optional[object]]:
        """Setup optimizer and learning rate scheduler.
        
        Args:
            train_dataloader: Training data loader.
            num_epochs: Number of training epochs.
            
        Returns:
            Tuple of (optimizer, scheduler).
        """
        # Optimizer
        optimizer_name = self.config.get('optimizer', 'AdamW')
        learning_rate = self.config.get('learning_rate', 2e-5)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # Scheduler
        scheduler_name = self.config.get('scheduler', 'linear')
        warmup_steps = self.config.get('warmup_steps', 100)
        
        if scheduler_name == 'linear':
            total_steps = len(train_dataloader) * num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            scheduler = None
        
        logger.info(f"Setup optimizer: {optimizer_name}, scheduler: {scheduler_name}")
        return optimizer, scheduler
    
    def train_epoch(
        self,
        train_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object]
    ) -> float:
        """Train for one epoch.
        
        Args:
            train_dataloader: Training data loader.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            
        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.config.get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        logger.info(f"Epoch {self.current_epoch} - Average loss: {avg_loss:.4f}")
        return avg_loss
    
    def evaluate(
        self,
        val_dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on validation set.
        
        Args:
            val_dataloader: Validation data loader.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_attention_masks = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                # Store for metrics calculation
                total_loss += loss.item()
                all_predictions.append(logits)
                all_labels.append(labels)
                all_attention_masks.append(attention_mask)
        
        # Concatenate all predictions
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(
            all_predictions, all_labels, all_attention_masks
        )
        
        # Add loss
        metrics['val_loss'] = total_loss / len(val_dataloader)
        
        # Calculate calibration metrics
        calibration_metrics = self.metrics.calculate_calibration_metrics(
            all_predictions, all_labels, all_attention_masks
        )
        metrics.update(calibration_metrics)
        
        self.val_metrics.append(metrics)
        
        logger.info(f"Validation metrics: F1={metrics['f1']:.4f}, "
                   f"Precision={metrics['precision']:.4f}, "
                   f"Recall={metrics['recall']:.4f}")
        
        return metrics
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_epochs: int,
        checkpoint_dir: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_dataloader: Training data loader.
            val_dataloader: Validation data loader.
            num_epochs: Number of training epochs.
            checkpoint_dir: Directory to save checkpoints.
            
        Returns:
            Training history.
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        # Setup optimizer and scheduler
        optimizer, scheduler = self.setup_optimizer_and_scheduler(
            train_dataloader, num_epochs
        )
        
        # Early stopping
        early_stopping = self.config.get('early_stopping', {})
        patience = early_stopping.get('patience', 3)
        min_delta = early_stopping.get('min_delta', 0.001)
        best_f1 = 0.0
        patience_counter = 0
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(train_dataloader, optimizer, scheduler)
            
            # Evaluate
            val_metrics = self.evaluate(val_dataloader)
            val_f1 = val_metrics['f1']
            
            # Check for improvement
            if val_f1 > best_f1 + min_delta:
                best_f1 = val_f1
                patience_counter = 0
                
                # Save best model
                if checkpoint_dir:
                    self.save_checkpoint(checkpoint_dir, is_best=True)
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping.get('enabled', True) and patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Save final model
        if checkpoint_dir:
            self.save_checkpoint(checkpoint_dir, is_best=False)
        
        logger.info(f"Training completed. Best F1: {best_f1:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
    
    def save_checkpoint(self, checkpoint_dir: str, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            checkpoint_dir: Directory to save checkpoint.
            is_best: Whether this is the best model.
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_f1': self.best_f1,
            'config': self.config
        }
        
        if is_best:
            checkpoint_path = Path(checkpoint_dir) / 'best_model.pt'
        else:
            checkpoint_path = Path(checkpoint_dir) / f'epoch_{self.current_epoch}.pt'
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_f1 = checkpoint['best_f1']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
