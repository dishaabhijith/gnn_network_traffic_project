# Graph Neural Networks for Predictive Analysis of Network Traffic

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-IEEE-blue)](link-to-paper)

> **A Comprehensive Framework for Real-Time Anomaly Detection and Traffic Forecasting**

This repository contains the implementation of a novel Graph Neural Network (GNN) framework for predictive analysis of network traffic, achieving **94.2% accuracy** in traffic prediction and anomaly detection with significantly reduced false positive rates of **3.5-6.5%**.

## Key Features

- **Hybrid Temporal GCN-LSTM Architecture**: Combines spatial graph learning with temporal sequence modeling
- **KNN-based Graph Augmentation**: Captures both observed communications and latent node similarities
- **Real-time Processing**: Web-based deployment with real-time inference capabilities
- **Superior Performance**: 5.5% improvement over state-of-the-art Graph WaveNet
- **Low False Positives**: Maintains 3.5-6.5% false positive rates across different network configurations
- **Scalable Design**: Supports networks from 50 to 1000+ nodes

## Performance Highlights

| Method | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|--------|----------|-----------|---------|----------|---------|
| **Proposed GCN-LSTM** | **94.2%** | **92.8%** | **91.5%** | **92.1%** | **0.941** |
| Graph WaveNet | 89.3% | 87.1% | 88.2% | 87.6% | 0.889 |
| Standard LSTM | 86.7% | 84.3% | 85.9% | 85.1% | 0.863 |
| Random Forest | 83.1% | 81.7% | 82.4% | 82.0% | 0.828 |

## Architecture Overview

```
Network Traffic Data → Graph Construction → GCN Layers → LSTM → Predictions
                              ↓
                    [Observed Edges + KNN Edges]
                              ↓
                    [Spatial Learning + Temporal Modeling]
                              ↓
                    [Traffic Forecasting + Anomaly Detection]
```

### Mathematical Foundation

The hybrid architecture combines Graph Convolutional Networks with LSTM:

```
H^(1)_t = ReLU(AX_t W^(1))
H^(2)_t = ReLU(AH^(1)_t W^(2))
O_t = LSTM(H^(2)_t)
Y_t = O_t W^(out) + b^(out)
```

## Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Required packages
pip install -r requirements.txt
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/gnn-network-traffic-analysis.git
cd gnn-network-traffic-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Demo

```python
from src.model import TemporalGCNLSTM
from src.data_preprocessing import NetworkDataProcessor
from src.utils import load_config

# Load configuration
config = load_config('config/default.yaml')

# Initialize model
model = TemporalGCNLSTM(
    node_features=config['node_features'],
    hidden_dim=config['hidden_dim'],
    num_classes=config['num_classes']
)

# Process your network data
processor = NetworkDataProcessor(config)
graph_data = processor.process_network_logs('data/network_logs.csv')

# Train the model
model.fit(graph_data)

# Make predictions
predictions = model.predict(new_graph_data)
```

## Project Structure

```
gnn-network-traffic-analysis/
├── src/
│   ├── model/
│   │   ├── gcn_lstm.py          # Main GCN-LSTM architecture
│   │   ├── layers.py            # Custom GNN layers
│   │   └── attention.py         # Attention mechanisms
│   ├── data/
│   │   ├── preprocessing.py     # Data preprocessing pipeline
│   │   ├── graph_builder.py     # Graph construction utilities
│   │   └── augmentation.py      # KNN-based augmentation
│   ├── training/
│   │   ├── trainer.py           # Training pipeline
│   │   ├── losses.py            # Custom loss functions
│   │   └── metrics.py           # Evaluation metrics
│   ├── web_app/
│   │   ├── app.py              # Flask web application
│   │   ├── static/             # CSS, JS, images
│   │   └── templates/          # HTML templates
│   └── utils/
│       ├── config.py           # Configuration management
│       ├── visualization.py    # Network visualization
│       └── logger.py           # Logging utilities
├── data/
│   ├── raw/                    # Raw network logs
│   ├── processed/              # Preprocessed datasets
│   └── examples/               # Sample datasets
├── config/
│   ├── default.yaml            # Default configuration
│   ├── model_configs/          # Model-specific configs
│   └── deployment.yaml         # Deployment settings
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_development.ipynb
│   ├── 03_evaluation.ipynb
│   └── 04_visualization.ipynb
├── tests/
│   ├── test_model.py
│   ├── test_preprocessing.py
│   └── test_integration.py
├── docs/
│   ├── architecture.md
│   ├── deployment_guide.md
│   └── api_reference.md
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
```

## Configuration

The framework supports flexible configuration through YAML files:

```yaml
# config/default.yaml
model:
  gcn_layers: 2
  gcn_hidden_dim: 64
  lstm_hidden_dim: 128
  dropout: 0.2
  learning_rate: 0.001

data:
  window_size: 10
  knn_neighbors: 5
  edge_threshold: 0.1
  normalization: "standard"

training:
  epochs: 100
  batch_size: 32
  validation_split: 0.2
  early_stopping: true
```

## Data Format

### Input Network Logs

Your network traffic data should be in CSV format with the following structure:

```csv
timestamp,source_ip,dest_ip,bytes_sent,bytes_received,protocol,port
2024-01-01 00:00:01,192.168.1.10,192.168.1.20,1024,2048,TCP,80
2024-01-01 00:00:02,192.168.1.15,192.168.1.25,512,1024,UDP,53
```

### Preprocessing Pipeline

The framework automatically:
1. **Temporal Windowing**: Aggregates traffic into fixed-duration intervals
2. **Graph Construction**: Creates nodes (IP addresses) and edges (communications)
3. **KNN Augmentation**: Adds edges based on feature similarity
4. **Feature Normalization**: Standardizes traffic volume features

## Web Interface

Launch the real-time monitoring dashboard:

```bash
# Start the web application
python src/web_app/app.py

# Access the dashboard
# Open http://localhost:5000 in your browser
```

### Dashboard Features

- **Real-time Traffic Visualization**: Live network topology and traffic flows
- **Anomaly Alerts**: Immediate notifications for detected anomalies
- **Prediction Charts**: Traffic forecasting with confidence intervals
- **Network Statistics**: Comprehensive network health metrics
- **Interactive Controls**: Adjustable thresholds and time windows

## Training Your Model

### Basic Training

```python
from src.training.trainer import NetworkTrafficTrainer

trainer = NetworkTrafficTrainer(config_path='config/default.yaml')
trainer.train(
    train_data='data/processed/train_data.pkl',
    val_data='data/processed/val_data.pkl'
)
```

### Advanced Training Options

```python
# Custom training with hyperparameter tuning
trainer = NetworkTrafficTrainer(config)
best_params = trainer.hyperparameter_search(
    param_space={
        'learning_rate': [0.001, 0.01, 0.1],
        'hidden_dim': [64, 128, 256],
        'dropout': [0.1, 0.2, 0.3]
    }
)

# Train with best parameters
trainer.train_with_params(best_params)
```

## Evaluation and Metrics

### Comprehensive Evaluation

```python
from src.training.evaluator import ModelEvaluator

evaluator = ModelEvaluator(model, test_data)
results = evaluator.comprehensive_evaluation()

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Precision: {results['precision']:.2%}")
print(f"Recall: {results['recall']:.2%}")
print(f"F1-Score: {results['f1_score']:.2%}")
```

### Anomaly Detection Metrics

- **True Positive Rate**: 91.5%
- **False Positive Rate**: 3.5-6.5%
- **Detection Latency**: <100ms for real-time alerts
- **Scalability**: Tested on networks up to 1000+ nodes

## Deployment Guide

### Docker Deployment

```dockerfile
# Build the container
docker build -t gnn-network-monitor .

# Run with GPU support
docker run --gpus all -p 5000:5000 gnn-network-monitor
```

### Production Considerations

- **Scalability**: Linear computational growth with network size
- **Memory Requirements**: 45-524 MB depending on network size
- **Inference Time**: 23-413 ms per prediction
- **Throughput**: 2.4-42.7 predictions/second

### Performance Optimization

For large networks (1000+ nodes):
- Consider distributed processing
- Implement model quantization
- Use batch inference for improved throughput

## Experiments and Results

### Ablation Study

| Model Variant | Accuracy | Performance Impact |
|---------------|----------|-------------------|
| Full Model | 94.2% | - |
| w/o KNN Edges | 91.8% | -2.4% |
| w/o LSTM | 89.1% | -5.1% |
| w/o GCN | 86.7% | -7.5% |

### Scalability Analysis

| Network Size | Inference Time | Memory Usage | Scalability |
|--------------|----------------|--------------|-------------|
| 50 nodes | 23.4 ms | 45.2 MB | Excellent |
| 200 nodes | 67.8 ms | 126.8 MB | Good |
| 500 nodes | 189.3 ms | 287.4 MB | Acceptable |
| 1000+ nodes | 412.7 ms | 523.9 MB | Requires Optimization |

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/

# Pre-commit hooks
pre-commit install
```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{disha2024gnn,
  title={Graph Neural Networks for Predictive Analysis of Network Traffic: A Comprehensive Framework for Real-Time Anomaly Detection and Traffic Forecasting},
  author={Disha, A. and Marpuri, Mahika},
  journal={IEEE Conference Proceedings},
  year={2024},
  publisher={IEEE}
}
```

## Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api_reference.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Troubleshooting](docs/troubleshooting.md)

## Known Issues

- Large networks (1000+ nodes) require optimization for real-time processing
- Memory usage grows quadratically with network size
- Limited support for dynamic topology changes during inference

## Future Roadmap

- [ ] Distributed processing for large-scale networks
- [ ] Dynamic graph adaptation for changing topologies
- [ ] Explainable AI features for anomaly interpretation
- [ ] Multi-modal data integration (logs, metrics, threat intel)
- [ ] Federated learning capabilities
- [ ] Advanced visualization and reporting features

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/gnn-network-traffic-analysis/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/gnn-network-traffic-analysis/discussions)
- **Email**: dishaa.is24@rvce.edu.in, mahikamarpuri.is23@rvce.edu.in

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- R V College of Engineering, Bengaluru
- PyTorch Geometric community
- Graph neural network research community
- Open-source contributors

---

**⭐ If you find this project useful, please consider giving it a star!**
