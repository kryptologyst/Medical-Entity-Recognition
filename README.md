# Medical Entity Recognition

A production-ready implementation of Medical Entity Recognition (NER) using BioBERT for extracting structured medical concepts from clinical text.

## ⚠️ IMPORTANT DISCLAIMER

**This software is for RESEARCH AND EDUCATIONAL PURPOSES ONLY.**

- **NOT FOR CLINICAL USE**: This tool is not intended for clinical diagnosis, treatment, or medical decision-making
- **NOT MEDICAL ADVICE**: Results should not be used as medical advice or recommendations
- **RESEARCH DEMO**: This is a demonstration of AI capabilities for research and educational purposes
- **CLINICIAN SUPERVISION**: Always consult qualified healthcare professionals for medical decisions

## Overview

This project implements a state-of-the-art Named Entity Recognition system specifically designed for medical text. It can identify and extract:

- **Diseases**: hypertension, diabetes, pneumonia, etc.
- **Chemicals/Medications**: metformin, insulin, aspirin, etc.
- **Symptoms**: chest pain, fatigue, dizziness, etc.
- **Procedures**: blood test, CT scan, surgery, etc.

## Features

### Core Capabilities
- **BioBERT-based NER**: Leverages pre-trained biomedical language models
- **Multiple Model Support**: BioBERT, ClinicalBERT, and SciSpaCy integration
- **Synthetic Data Generation**: Creates realistic clinical notes for training/testing
- **Comprehensive Evaluation**: Token-level and entity-level metrics with calibration

### Privacy & Compliance
- **De-identification**: Automatic PHI detection and redaction
- **Privacy Safeguards**: No PHI logging, configurable privacy controls
- **Compliance Ready**: Built-in privacy utilities and audit trails

### Explainability & Safety
- **Attention Visualization**: Visualize model attention patterns
- **Entity Attribution**: Understand which tokens contribute to predictions
- **Confidence Calibration**: Reliable uncertainty quantification
- **Safety Filters**: Built-in safeguards and warnings

### Modern ML Stack
- **PyTorch 2.x**: Latest deep learning framework
- **Transformers**: State-of-the-art NLP models
- **Hydra Configuration**: Flexible, hierarchical configuration management
- **Comprehensive Logging**: Structured logging with privacy protection

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- CUDA/MPS support (optional but recommended)

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/kryptologyst/Medical-Entity-Recognition.git
cd Medical-Entity-Recognition
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download SciSpaCy models** (optional):
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_ner_bc5cdr_md-0.5.1.tar.gz
```

## Quick Start

### 1. Generate Synthetic Dataset
```bash
python scripts/generate_data.py --config configs/data/synthetic.yaml --output data/synthetic_dataset.json
```

### 2. Train Model
```bash
python scripts/train.py --config configs/config.yaml --output_dir outputs --checkpoint_dir checkpoints
```

### 3. Run Demo
```bash
streamlit run demo/app.py
```

### 4. Evaluate Model
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --test_data data/test.json
```

## Usage Examples

### Basic NER Prediction
```python
from src.models.biobert_model import BioBERTNERModel

# Initialize model
config = {'model_name': 'dmis-lab/biobert-base-cased-v1.1'}
model = BioBERTNERModel(config)

# Predict entities
text = "Patient diagnosed with hypertension. Prescribed metformin."
entities = model.predict([text])[0]

for entity in entities:
    print(f"{entity['text']} -> {entity['label']}")
```

### De-identification
```python
from src.utils.deid import DeIdentifier

# Initialize de-identifier
deid = DeIdentifier(enable_deid=True)

# De-identify text
text = "Patient John Doe, MRN 12345, diagnosed with diabetes."
deidentified_text, phi_entities = deid.deidentify_text(text)

print(f"Original: {text}")
print(f"De-identified: {deidentified_text}")
```

### Explainability
```python
from src.utils.explainability import ModelExplainer

# Initialize explainer
explainer = ModelExplainer(model, tokenizer)

# Generate explanations
explanations = explainer.explain_prediction(text, entities)
```

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/config.yaml`: Main configuration
- `configs/model/biobert.yaml`: Model-specific settings
- `configs/data/synthetic.yaml`: Dataset configuration
- `configs/training/default.yaml`: Training parameters

### Example Configuration
```yaml
# Model configuration
model:
  model_name: "dmis-lab/biobert-base-cased-v1.1"
  max_length: 512
  num_labels: 9
  dropout: 0.1

# Training configuration
training:
  epochs: 10
  batch_size: 16
  learning_rate: 2e-5
  early_stopping:
    enabled: true
    patience: 3

# Privacy configuration
privacy:
  enable_deid: true
  redact_phi: true
  log_phi: false
```

## Project Structure

```
medical_entity_recognition/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   ├── data/                     # Data processing
│   ├── train/                    # Training pipeline
│   ├── metrics/                  # Evaluation metrics
│   └── utils/                    # Utilities
├── configs/                      # Configuration files
├── scripts/                      # Training/evaluation scripts
├── demo/                         # Streamlit demo app
├── tests/                        # Unit tests
├── assets/                       # Generated visualizations
├── data/                         # Data directory
├── outputs/                      # Training outputs
├── checkpoints/                  # Model checkpoints
└── requirements.txt               # Dependencies
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

### Token-level Metrics
- **Accuracy**: Overall token classification accuracy
- **Precision/Recall/F1**: Weighted, micro, and macro averages
- **Per-entity Metrics**: Individual performance for each entity type

### Entity-level Metrics
- **Entity Precision/Recall/F1**: Exact match evaluation
- **Entity-level Analysis**: Per-entity-type performance

### Calibration Metrics
- **Expected Calibration Error (ECE)**: Calibration quality
- **Brier Score**: Probability calibration
- **Confidence Analysis**: Reliability of predictions

## Demo Application

The Streamlit demo provides an interactive interface for:

- **Text Input**: Enter or select sample clinical text
- **Entity Detection**: Real-time entity recognition
- **Visualization**: Highlighted entities with confidence scores
- **Privacy Controls**: Optional de-identification
- **Explainability**: Attention visualization and entity attribution

### Running the Demo
```bash
streamlit run demo/app.py
```

## Privacy and Compliance

### De-identification Features
- **PHI Detection**: Automatic detection of personally identifiable information
- **Redaction**: Configurable redaction of sensitive data
- **Audit Trails**: Logging of de-identification actions (without PHI)

### Privacy Controls
- **No PHI Logging**: Configurable to prevent PHI in logs
- **Data Minimization**: Only necessary data is processed
- **User Control**: Configurable privacy settings

### Compliance Notes
- **HIPAA Awareness**: Built with healthcare privacy in mind
- **Research Use**: Clearly marked as research/educational tool
- **No Clinical Claims**: Explicit disclaimers throughout

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific test categories:
```bash
pytest tests/test_models.py -v
pytest tests/test_data.py -v
pytest tests/test_utils.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code Style
- **Black**: Code formatting
- **Ruff**: Linting
- **Type Hints**: Required for all functions
- **Docstrings**: Google/NumPy style

## Limitations and Known Issues

### Model Limitations
- **Domain Specificity**: Trained on biomedical text, may not generalize to other domains
- **Language Support**: Currently English-only
- **Entity Types**: Limited to predefined medical entity types

### Technical Limitations
- **Context Length**: Limited by transformer model context window
- **Computational Requirements**: Requires GPU for optimal performance
- **Model Size**: Large models require significant memory

### Clinical Limitations
- **Not Diagnostic**: Results should not be used for clinical diagnosis
- **Accuracy Limitations**: May miss or misclassify entities
- **Bias Considerations**: Models may reflect training data biases

## Citation

If you use this code in your research, please cite:

```bibtex
@software{medical_ner_2025,
  title={Medical Entity Recognition: A BioBERT-based System},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Medical-Entity-Recognition},
  note={Research demonstration tool - not for clinical use}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **BioBERT**: Pre-trained biomedical language models
- **Transformers**: Hugging Face transformer library
- **SciSpaCy**: Biomedical NLP tools
- **PyTorch**: Deep learning framework

---

**Remember**: This is a research demonstration tool. Always consult qualified healthcare professionals for medical decisions.
# Medical-Entity-Recognition
