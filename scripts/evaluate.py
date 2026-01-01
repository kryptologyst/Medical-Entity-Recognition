"""Evaluation script for trained models."""

import argparse
import json
import logging
import torch
from pathlib import Path
from typing import Dict, Any

from src.models.biobert_model import BioBERTNERModel
from src.data.synthetic_dataset import SyntheticNERDataset
from src.train.trainer import NERTrainer
from src.utils import get_device, load_config, create_directories

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Evaluate trained model."""
    parser = argparse.ArgumentParser(description='Evaluate Medical Entity Recognition Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, default='data/test.json',
                       help='Path to test dataset')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    create_directories([args.output_dir])
    
    # Get device
    device = get_device(args.device)
    
    logger.info("Loading model checkpoint...")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Initialize model
    model_config = config.get('model', {})
    model = BioBERTNERModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info("Loading test dataset...")
    
    # Load test dataset
    if Path(args.test_data).exists():
        dataset = SyntheticNERDataset(config.get('data', {}))
        test_notes = dataset.load_dataset(args.test_data)
    else:
        logger.warning(f"Test data file {args.test_data} not found. Generating synthetic test data...")
        dataset = SyntheticNERDataset(config.get('data', {}))
        test_notes = dataset.generate_dataset()
        # Use only a subset for testing
        test_notes = test_notes[:100]
    
    logger.info(f"Evaluating on {len(test_notes)} test samples...")
    
    # Initialize trainer for evaluation
    training_config = config.get('training', {})
    trainer = NERTrainer(model, training_config, device)
    
    # Evaluate model
    test_metrics = trainer.evaluate(test_notes)
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info(f"F1 Score: {test_metrics['f1']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Per-entity metrics
    entity_types = config.get('model', {}).get('entity_types', ['DISEASE', 'CHEMICAL', 'SYMPTOM', 'PROCEDURE'])
    for entity_type in entity_types:
        f1_key = f'{entity_type}_f1'
        if f1_key in test_metrics:
            logger.info(f"{entity_type} F1: {test_metrics[f1_key]:.4f}")
    
    # Calibration metrics
    if 'ece' in test_metrics:
        logger.info(f"Expected Calibration Error: {test_metrics['ece']:.4f}")
    if 'brier_score' in test_metrics:
        logger.info(f"Brier Score: {test_metrics['brier_score']:.4f}")
    
    # Save results
    results = {
        'checkpoint_path': args.checkpoint,
        'test_data_path': args.test_data,
        'num_test_samples': len(test_notes),
        'metrics': test_metrics,
        'config': config
    }
    
    results_path = Path(args.output_dir) / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {results_path}")


if __name__ == '__main__':
    main()
