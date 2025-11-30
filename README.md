# Research Compass - GNN Paper Classification

Graph Neural Network based system for classifying research papers using PyTorch Geometric.

## Overview

This project uses Graph Neural Networks (GAT, GCN, GraphSAGE) to predict research paper topics. You can upload PDFs or paste text, and the models will classify them into appropriate categories. The system supports two datasets: OGB arXiv (169K CS papers) and AMiner (10K authors).

**Key features:**
- Multiple GNN architectures (GAT, GCN, GraphSAGE)
- PDF upload and text extraction
- Interactive knowledge graph visualization
- Two datasets: OGB arXiv (40 CS topics) and AMiner (8 research fields)
- Real-time predictions with confidence scores

## Model Performance

### OGB arXiv Dataset (169K CS Papers, 40 Topics)

| Model | Test Accuracy | Link Prediction AUC |
|-------|---------------|---------------------|
| GAT | 53.4% | 78.0% |
| GCN | 50.3% | 90.1% |
| GraphSAGE | 52.3% | 79.6% |

All three models perform similarly on this dataset (50-57% accuracy). The high-quality dataset with balanced classes and dense citation network means architecture choice doesn't matter much here.

### AMiner Dataset (10K Authors, 8 Fields)

| Model | Test Accuracy | Link Prediction AUC |
|-------|---------------|---------------------|
| GraphSAGE | 71.0% | 71.0% |
| GAT | 42.8% | 58.7% |
| GCN | 28.0% | 50.4% |

**Important:** For AMiner, use GraphSAGE only. GAT and GCN perform poorly due to class imbalance and sparse features.

## Installation

```bash
git clone https://github.com/Apc0015/Research_Compass_GNN.git
cd Research_Compass_GNN
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Training Your Own Models

Training notebooks are in the `notebooks/` folder:

**OGB arXiv:**
```bash
jupyter notebook notebooks/GNN_OGB.ipynb
```
- Training time: 30-60 minutes (GPU recommended)
- Expected accuracy: 50-57%
- All models perform similarly

**AMiner:**
```bash
jupyter notebook notebooks/GNN_AMiner.ipynb
```
- Training time: 20-40 minutes (GPU recommended)
- Expected accuracy: 71% (GraphSAGE only)
- Note: GAT and GCN have much lower accuracy (28-43%)

Models save automatically to `saved_models/` directory.

## Project Structure

```
Research_Compass_GNN/
├── app.py                   # Streamlit web app
├── .gitignore
├── saved_models/
│   ├── OGB_models.pt       # OGB trained models
│   └── aminer_models.pt    # AMiner trained models
├── notebooks/
│   ├── GNN_OGB.ipynb       # OGB training notebook
│   └── GNN_AMiner.ipynb    # AMiner training notebook
├── requirements.txt
├── packages.txt
└── README.md

data/                        # Auto-downloaded (not in git)
```

## How It Works

1. Upload a PDF or paste text
2. Select dataset (OGB arXiv for papers, AMiner for authors)
3. Choose model (GraphSAGE recommended)
4. Get predictions with confidence scores
5. View knowledge graph visualization

## Model Architectures

**Graph Attention Network (GAT)**
- OGB arXiv: 3 layers, 128 hidden dims, 4 attention heads, batch norm
- AMiner: 2 layers, 256 hidden dims, 4 attention heads, no batch norm
- Best for OGB arXiv (53.4%)

**Graph Convolutional Network (GCN)**
- OGB arXiv: 3 layers, 128 hidden dims, batch norm
- AMiner: 2 layers, 256 hidden dims, no batch norm
- Best for OGB link prediction (90.1% AUC)

**GraphSAGE**
- OGB arXiv: 3 layers, 128 hidden dims, batch norm
- AMiner: 2 layers, 256 hidden dims, no batch norm
- Best overall (works well on both datasets)

## Datasets

**OGB arXiv**
- 169,343 Computer Science papers (1993-2020)
- 40 categories (cs.AI, cs.LG, cs.CV, etc.)
- 1,166,243 citation edges
- 128-dim features from abstracts

**AMiner**
- 10,000 authors
- 8 research fields
- ~120K co-authorship edges
- 136-dim features (128 base + 8 class embeddings)

## Troubleshooting

**Low prediction confidence?**
- Use GraphSAGE for AMiner dataset
- Provide more text (500+ words)
- Include abstract and introduction

**Models not loading?**
- Run training notebooks first
- Check `saved_models/` directory exists
- Verify PyTorch 2.0+ installed

**AMiner accuracy seems low?**
- This is expected due to dataset quality
- Only GraphSAGE achieves 71% accuracy
- GAT/GCN are included for comparison only

## Technical Stack

- Streamlit - web framework
- PyTorch 2.0+ - deep learning
- PyTorch Geometric - GNN library
- OGB - graph benchmark datasets
- Plotly - visualizations
- NetworkX - graph algorithms
- PyPDF2 - PDF extraction
- scikit-learn - preprocessing

## References

**Datasets:**
- OGB arXiv: Open Graph Benchmark (Hu et al., NeurIPS 2020)
- AMiner: AMiner Dataset (Tang et al., KDD 2008)

**Models:**
- GAT: Graph Attention Networks (Veličković et al., ICLR 2018)
- GCN: Semi-Supervised Classification with GCNs (Kipf & Welling, ICLR 2017)
- GraphSAGE: Inductive Representation Learning (Hamilton et al., NeurIPS 2017)

## License

MIT License

## Contributing

Pull requests welcome. Please open an issue first to discuss changes.

## Contact

For issues, use the GitHub issue tracker: https://github.com/Apc0015/Research_Compass_GNN/issues
