# HQ-LSTM

## Hybrid Quantum-Classical LSTM for Sentiment Analysis

A research implementation of a Hybrid Quantum-Classical LSTM architecture for sentiment classification. This project compares a quantum-inspired LSTM approach against a classical LSTM baseline, with noise robustness experiments.

## Overview

This project implements two models:

1. **Classical LSTM Baseline** - A standard bidirectional LSTM for sentiment classification
2. **HQ-LSTM (Hybrid Quantum-Classical LSTM)** - A quantum-inspired LSTM using custom LSTM cells with noise injection for robustness testing

## Features

- Classical LSTM baseline with configurable layers and bidirectional support
- Quantum-inspired LSTM cell implementation
- Hybrid quantum-classical architecture with noise injection
- Noise robustness experiments at multiple noise levels
- Training with learning rate scheduling and gradient clipping
- Comprehensive visualization of results

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
HQ-LSTM/
|-- hq_lstm.py          # Main model implementations and training code
|-- requirements.txt    # Python dependencies
|-- README.md          # This file
|-- LICENSE            # MIT License
|-- .gitignore         # Python gitignore
```

## Usage

### Running the Models

```bash
python hq_lstm.py
```

This will:
1. Create synthetic sentiment datasets (train/val/test)
2. Train both the HQ-LSTM and Classical LSTM models
3. Run noise robustness experiments
4. Display training curves and comparison results

### Model Configuration

The following hyperparameters can be adjusted in `hq_lstm.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| VOCAB_SIZE | 5000 | Vocabulary size |
| EMBED_DIM | 128 | Embedding dimension |
| HIDDEN_DIM | 128 | Hidden layer dimension |
| NUM_LAYERS | 2 | Number of LSTM layers |
| DROPOUT | 0.2 | Dropout rate |
| EPOCHS | 5 | Training epochs |
| LEARNING_RATE | 0.001 | Learning rate |

## Architecture Details

### Classical LSTM
- Uses PyTorch's built-in `nn.LSTM` module
- Supports bidirectional configuration
- FC layer with dropout and ReLU activation

### HQ-LSTM
- Custom quantum-inspired LSTM cells with forget, input, output, and cell gates
- Hybrid architecture that can toggle quantum mode on/off
- Built-in noise injection for robustness testing
- Noise scale parameter for controlled perturbation

### Noise Robustness
The HQ-LSTM model includes a noise injection mechanism that adds Gaussian noise to hidden states during training and evaluation. Experiments are conducted at noise levels: 0.0, 0.1, 0.2, 0.3, and 0.5.

## Results

The experiments produce:
- Training loss curves for both models
- Validation accuracy per epoch
- Noise robustness comparison across noise levels
- Side-by-side model comparison

## Technologies Used

- **PyTorch** - Deep learning framework
- **torchtext** - Text data processing
- **NumPy** - Numerical computations
- **Matplotlib** - Visualization
- **scikit-learn** - Utilities
- **tqdm** - Progress bars

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was developed as a research implementation for exploring hybrid quantum-classical architectures in NLP tasks.
