"""Data generation script for synthetic clinical notes."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

from src.data.synthetic_dataset import SyntheticNERDataset
from src.utils import load_config, create_directories

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Generate synthetic dataset."""
    parser = argparse.ArgumentParser(description='Generate Synthetic Clinical Notes Dataset')
    parser.add_argument('--config', type=str, default='configs/data/synthetic.yaml',
                       help='Path to data configuration file')
    parser.add_argument('--output', type=str, default='data/synthetic_dataset.json',
                       help='Output path for generated dataset')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to generate (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override num_samples if provided
    if args.num_samples is not None:
        config['num_samples'] = args.num_samples
    
    # Create output directory
    output_path = Path(args.output)
    create_directories([output_path.parent])
    
    # Initialize dataset generator
    dataset = SyntheticNERDataset(config)
    
    # Generate dataset
    logger.info(f"Generating {config['num_samples']} synthetic clinical notes...")
    notes = dataset.generate_dataset()
    
    # Save dataset
    dataset.save_dataset(notes, args.output)
    
    logger.info(f"Dataset saved to {args.output}")
    logger.info(f"Generated {len(notes)} clinical notes")
    
    # Print sample statistics
    total_entities = sum(len(note.entities) for note in notes)
    entity_types = {}
    for note in notes:
        for entity in note.entities:
            entity_types[entity.label] = entity_types.get(entity.label, 0) + 1
    
    logger.info(f"Total entities: {total_entities}")
    logger.info(f"Entity type distribution: {entity_types}")


if __name__ == '__main__':
    main()
