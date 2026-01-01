#!/usr/bin/env python3
"""Main training script for medical entity recognition."""

import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from src.models.biobert_model import BioBERTNERModel
from src.data.synthetic_dataset import SyntheticNERDataset
from src.train.trainer import NERTrainer
from src.utils import set_seed, get_device, create_directories, load_config
from src.utils.deid import DeIdentifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_data_loaders(
    dataset: SyntheticNERDataset,
    tokenizer,
    config: Dict[str, Any],
    batch_size: int = 16
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders.
    
    Args:
        dataset: Dataset to split.
        tokenizer: Tokenizer for encoding text.
        config: Configuration dictionary.
        batch_size: Batch size for data loaders.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    # Generate dataset
    notes = dataset.generate_dataset()
    
    # Split dataset
    train_split = config.get('train_split', 0.7)
    val_split = config.get('val_split', 0.15)
    test_split = config.get('test_split', 0.15)
    
    total_size = len(notes)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    train_notes, val_notes, test_notes = random_split(
        notes, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Dataset split: Train={len(train_notes)}, Val={len(val_notes)}, Test={len(test_notes)}")
    
    # Create data loaders (simplified - in practice you'd need proper collation)
    train_loader = DataLoader(train_notes, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_notes, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_notes, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Medical Entity Recognition Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory for data files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, mps, cpu)')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directories
    create_directories([args.output_dir, args.checkpoint_dir, args.data_dir])
    
    # Get device
    device = get_device(args.device)
    
    # Initialize de-identifier
    deid_config = config.get('privacy', {})
    deidentifier = DeIdentifier(deid_config.get('enable_deid', True))
    
    logger.info("Starting Medical Entity Recognition Training")
    logger.info(f"Configuration: {config}")
    
    # Initialize dataset
    data_config = config.get('data', {})
    dataset = SyntheticNERDataset(data_config)
    
    # Initialize tokenizer
    model_config = config.get('model', {})
    tokenizer = AutoTokenizer.from_pretrained(model_config.get('model_name', 'dmis-lab/biobert-base-cased-v1.1'))
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, tokenizer, data_config
    )
    
    # Initialize model
    model = BioBERTNERModel(model_config)
    
    # Initialize trainer
    training_config = config.get('training', {})
    trainer = NERTrainer(model, training_config, device)
    
    # Train model
    num_epochs = training_config.get('epochs', 10)
    history = trainer.train(
        train_loader, val_loader, num_epochs, args.checkpoint_dir
    )
    
    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    
    # Save results
    results = {
        'config': config,
        'history': history,
        'test_metrics': test_metrics,
        'model_info': {
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }
    
    results_path = Path(args.output_dir) / 'training_results.yaml'
    with open(results_path, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    logger.info(f"Training completed. Results saved to {results_path}")
    logger.info(f"Test F1 Score: {test_metrics['f1']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")


if __name__ == '__main__':
    main()
