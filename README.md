# Research Compass — GNN Paper Classification & Citation Link Prediction

[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-torch--geometric-3C8DBC.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-3%20passed-10B981.svg)](tests/)

**Research Compass** is a Graph Neural Network (GNN) platform I built to classify research papers into academic topics and predict citation links using graph structure. It trains and compares three major GNN architectures—**Graph Attention Networks (GAT)**, **Graph Convolutional Networks (GCN)**, and **GraphSAGE**—across two benchmark citation and co-authorship datasets.

**Live Demo:** [researchcompassgnn-bcvqiw3gxtrdgqdkxhv4ht.streamlit.app](https://researchcompassgnn-bcvqiw3gxtrdgqdkxhv4ht.streamlit.app/)

---

## What I Built

- **Comparative GNN Zoo:** Implemented and benchmarked GAT (with multi-head attention), GCN, and GraphSAGE for both node classification (topic prediction) and link prediction (citation recommendation).
- **Two Benchmark Datasets:**
  - **OGB arXiv:** 169,343 Computer Science papers, 40 topic categories, and 1,166,243 directed citation edges.
  - **AMiner:** 10,000 authors, 8 research fields, and ~120,000 co-authorship edges.
- **Interactive PDF & Text Ingestion:** Upload an academic paper PDF or paste raw text to extract TF-IDF/word features, project them into graph space, and predict topics with confidence scores.
- **Interactive Graph Visualizer:** Visualizes citation neighborhoods and co-authorship clusters in 2D using NetworkX, Plotly, and t-SNE embeddings.
- **Saved Model Checkpoints:** Serialized trained PyTorch models stored in `saved_models/` so the web interface runs immediate inference without retraining.

---

## Benchmark Results

Here are the actual results from evaluating all models on test splits:

### 1. OGB arXiv Dataset (169,343 Papers · 40 CS Categories)

| Model | Test Accuracy | Link Prediction AUC | Architecture Details |
|:---|:---:|:---:|:---|
| **GAT** | **53.4%** | 78.0% | 3 layers, 128 hidden channels, 4 attention heads, BatchNorm |
| **GraphSAGE** | 52.3% | 79.6% | 3 layers, 128 hidden channels, mean aggregator, BatchNorm |
| **GCN** | 50.3% | **90.1%** | 3 layers, 128 hidden channels, spectral convolution, BatchNorm |

*Takeaway:* On OGB arXiv, all three models perform within a tight 50–54% range. The dense citation network and balanced classes mean architectural differences have modest impact on node classification, but GCN excels on link prediction (90.1% AUC).

### 2. AMiner Co-Authorship Dataset (10,000 Authors · 8 Research Fields)

| Model | Test Accuracy | Link Prediction AUC | Architecture Details |
|:---|:---:|:---:|:---|
| **GraphSAGE** | **71.0%** | **71.0%** | 2 layers, 256 hidden channels, inductive neighbor sampling |
| **GAT** | 42.8% | 58.7% | 2 layers, 256 hidden channels, 4 attention heads |
| **GCN** | 28.0% | 50.4% | 2 layers, 256 hidden channels, spectral convolution |

*Takeaway:* AMiner has significant class imbalance and sparse feature representations. GCN and GAT struggle with over-smoothing and sparse neighborhoods, while **GraphSAGE significantly outperforms them (71.0% accuracy)** because its localized inductive sampling generalizes much better on sparse graphs.

---

## Project Structure

```
Research_Compass_GNN/
├── app.py                   # Streamlit web application & model inference
├── saved_models/
│   ├── OGB_models.pt        # Trained GAT, GCN, GraphSAGE weights for OGB arXiv
│   └── aminer_models.pt     # Trained weights for AMiner dataset
├── notebooks/
│   ├── GNN_OGB.ipynb        # Data preparation & training pipeline for OGB arXiv
│   └── GNN_AMiner.ipynb     # Training pipeline for AMiner dataset
├── tests/
│   └── test_model_checkpoints.py # Unit tests for model checkpoints & categories
├── requirements.txt         # Python dependencies
├── packages.txt             # Deployment system packages
├── pytest.ini               # Pytest configuration
└── README.md
```

---

## Quick Start

### 1. Try the Live App
You can test the deployed system directly at:  
[researchcompassgnn-bcvqiw3gxtrdgqdkxhv4ht.streamlit.app](https://researchcompassgnn-bcvqiw3gxtrdgqdkxhv4ht.streamlit.app/)

### 2. Run Locally

```bash
# Clone the repository
git clone https://github.com/Apc0015/Research_Compass_GNN.git
cd Research_Compass_GNN

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run unit tests
pytest tests/ -v

# Start the Streamlit app
streamlit run app.py
```

Open **`http://localhost:8501`** in your browser.

---

## Training the Models from Scratch

If you want to retrain the models, the notebooks in `notebooks/` contain the full training loops:

- **OGB arXiv:** `notebooks/GNN_OGB.ipynb` (takes ~30–60 minutes on GPU).
- **AMiner:** `notebooks/GNN_AMiner.ipynb` (takes ~20–40 minutes on GPU).

Trained checkpoints automatically save into `saved_models/`.

---

## Authors & Acknowledgments

- **Ayush Chhoker** - [Apc0015](https://github.com/Apc0015)
- **Datasets:** Open Graph Benchmark (OGB) arXiv (`Hu et al., NeurIPS 2020`) and AMiner Academic Network (`Tang et al., KDD 2008`).
- **GNN Architectures:** GAT (`Veličković et al., ICLR 2018`), GCN (`Kipf & Welling, ICLR 2017`), GraphSAGE (`Hamilton et al., NeurIPS 2017`).

---

## License

MIT License. See [LICENSE](LICENSE) for details.
