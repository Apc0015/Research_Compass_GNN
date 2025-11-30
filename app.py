"""
Research Compass - GNN Paper Classification System
Streamlit UI for predicting research paper topics using Graph Neural Networks
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from pathlib import Path
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import plotly.graph_objects as go
import networkx as nx
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Research Compass - GNN Paper Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 10px;
        height: 30px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model Definitions - AMiner (2 layers, no batch norm)
class GAT_AMiner(nn.Module):
    """Graph Attention Network for AMiner (2-layer)"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

class GCN_AMiner(nn.Module):
    """Graph Convolutional Network for AMiner (2-layer)"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

class GraphSAGE_AMiner(nn.Module):
    """GraphSAGE Network for AMiner (2-layer)"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)

# Model Definitions - OGB (3 layers, with batch norm)
class GAT(nn.Module):
    """Graph Attention Network for OGB (3-layer)"""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_channels * heads)
        self.conv3 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = self.bn2(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

class GCN(nn.Module):
    """Graph Convolutional Network for OGB (3-layer)"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

class GraphSAGE(nn.Module):
    """GraphSAGE Network for OGB (3-layer)"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

# arXiv CS category mapping (40 topics)
ARXIV_CATEGORIES = {
    0: "cs.AI - Artificial Intelligence",
    1: "cs.AR - Hardware Architecture",
    2: "cs.CC - Computational Complexity",
    3: "cs.CE - Computational Engineering",
    4: "cs.CG - Computational Geometry",
    5: "cs.CL - Computation and Language",
    6: "cs.CR - Cryptography and Security",
    7: "cs.CV - Computer Vision",
    8: "cs.CY - Computers and Society",
    9: "cs.DB - Databases",
    10: "cs.DC - Distributed Computing",
    11: "cs.DL - Digital Libraries",
    12: "cs.DM - Discrete Mathematics",
    13: "cs.DS - Data Structures and Algorithms",
    14: "cs.ET - Emerging Technologies",
    15: "cs.FL - Formal Languages",
    16: "cs.GL - General Literature",
    17: "cs.GR - Graphics",
    18: "cs.GT - Computer Science and Game Theory",
    19: "cs.HC - Human-Computer Interaction",
    20: "cs.IR - Information Retrieval",
    21: "cs.IT - Information Theory",
    22: "cs.LG - Machine Learning",
    23: "cs.LO - Logic in Computer Science",
    24: "cs.MA - Multiagent Systems",
    25: "cs.MM - Multimedia",
    26: "cs.MS - Mathematical Software",
    27: "cs.NA - Numerical Analysis",
    28: "cs.NE - Neural and Evolutionary Computing",
    29: "cs.NI - Networking and Internet Architecture",
    30: "cs.OH - Other Computer Science",
    31: "cs.OS - Operating Systems",
    32: "cs.PF - Performance",
    33: "cs.PL - Programming Languages",
    34: "cs.RO - Robotics",
    35: "cs.SC - Symbolic Computation",
    36: "cs.SD - Sound",
    37: "cs.SE - Software Engineering",
    38: "cs.SI - Social and Information Networks",
    39: "cs.SY - Systems and Control"
}

# AMiner research field mapping (8 topics, matching trained models)
AMINER_CATEGORIES = {
    0: "Machine Learning & AI",
    1: "Data Mining & Analytics",
    2: "Computer Vision & Graphics",
    3: "Natural Language Processing",
    4: "Databases & Information Systems",
    5: "Networks & Distributed Systems",
    6: "Software Engineering",
    7: "Security & Cryptography"
}

DATASET_INFO = {
    "OGB arXiv": {
        "categories": ARXIV_CATEGORIES,
        "model_file": "saved_models/OGB_models.pt",
        "num_classes": 40,
        "feature_dim": 128,
        "description": "169K CS papers from arXiv (1993-2020)",
        "task": "Paper topic classification"
    },
    "AMiner": {
        "categories": AMINER_CATEGORIES,
        "model_file": "saved_models/aminer_models.pt",
        "num_classes": 8,  # Updated from notebook (8 classes, not 10)
        "feature_dim": 136,  # 128 base + 8 class embeddings from notebook
        "description": "10K authors from AMiner network",
        "task": "Research field prediction"
    }
}

@st.cache_resource
def load_models(dataset_name="OGB arXiv"):
    """Load trained GNN models for the selected dataset"""
    device = torch.device('cpu')
    dataset_config = DATASET_INFO[dataset_name]
    model_path = Path(dataset_config['model_file'])
    
    if not model_path.exists():
        st.warning(f"Model file not found: {model_path}")
        return None, None, None, None, dataset_config
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None, dataset_config

    if dataset_name == "AMiner":
        feature_dim = checkpoint.get('num_features', dataset_config['feature_dim'])
        num_classes = checkpoint.get('num_classes', dataset_config['num_classes'])
        # AMiner uses 2-layer models without batch normalization
        gat = GAT_AMiner(feature_dim, 256, num_classes, heads=4, dropout=0.5)
        gcn = GCN_AMiner(feature_dim, 256, num_classes, dropout=0.5)
        sage = GraphSAGE_AMiner(feature_dim, 256, num_classes, dropout=0.5)
        try:
            gat.load_state_dict(checkpoint['gat_node'])
            gcn.load_state_dict(checkpoint['gcn_node'])
            sage.load_state_dict(checkpoint['sage_node'])
            gat.eval()
            gcn.eval()
            sage.eval()
            results_node = pd.DataFrame(checkpoint['results_node'])
            return gat, gcn, sage, results_node, {**dataset_config, 'feature_dim': feature_dim, 'num_classes': num_classes}
        except RuntimeError as e:
            st.error(f"Model weights could not be loaded due to a size mismatch. Please retrain your models for the current dataset configuration.\nError: {e}")
            return None, None, None, None, dataset_config
    else:  # OGB arXiv
        feature_dim = checkpoint.get('num_features', dataset_config['feature_dim'])
        num_classes = checkpoint.get('num_classes', dataset_config['num_classes'])
        # OGB uses 3-layer models with batch normalization
        gat = GAT(feature_dim, 128, num_classes, heads=4, dropout=0.5)
        gcn = GCN(feature_dim, 128, num_classes, dropout=0.5)
        sage = GraphSAGE(feature_dim, 128, num_classes, dropout=0.5)
        try:
            gat.load_state_dict(checkpoint['gat'])
            gcn.load_state_dict(checkpoint['gcn'])
            sage.load_state_dict(checkpoint['sage'])
            gat.eval()
            gcn.eval()
            sage.eval()
            results = checkpoint['results']['node_classification'] if 'results' in checkpoint and 'node_classification' in checkpoint['results'] else None
            results_df = pd.DataFrame(results) if results is not None else None
            return gat, gcn, sage, results_df, {**dataset_config, 'feature_dim': feature_dim, 'num_classes': num_classes}
        except RuntimeError as e:
            st.error(f"Model weights could not be loaded due to a size mismatch. Please retrain your models for the current dataset configuration.\nError: {e}")
            return None, None, None, None, dataset_config

@st.cache_resource
def load_dataset_for_features(dataset_name="OGB arXiv"):
    """Load dataset to get feature statistics"""
    dataset_config = DATASET_INFO[dataset_name]
    
    if dataset_name == "OGB arXiv":
        try:
            from ogb.nodeproppred import PygNodePropPredDataset
            import torch.serialization
            import os
            from torch_geometric.data import Data
            from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
            from torch_geometric.data.storage import GlobalStorage, EdgeStorage, NodeStorage
            
            torch.serialization.add_safe_globals([
                DataEdgeAttr, DataTensorAttr, Data,
                GlobalStorage, EdgeStorage, NodeStorage
            ])
            
            # Set environment variable to avoid fork-related crashes on macOS
            os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
            
            # Check if dataset is already processed to avoid unnecessary downloads
            dataset_path = os.path.join('data', 'ogbn_arxiv', 'processed')
            if os.path.exists(dataset_path):
                try:
                    # Try to load directly from processed files if they exist
                    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/')
                    data = dataset[0]
                    st.success("Loaded OGB arXiv dataset with real citation network")
                    return data
                except Exception as e:
                    st.error(f"Error loading processed dataset: {e}")
                    st.error("Real citation network visualization requires the actual OGB arXiv dataset.")
                    return None
            else:
                st.info("Downloading OGB arXiv dataset... (This may take a few minutes)")
                try:
                    dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='data/')
                    data = dataset[0]
                    st.success("Successfully downloaded and loaded OGB arXiv dataset!")
                    return data
                except Exception as e:
                    st.error(f"Failed to download dataset: {e}")
                    return None
        except Exception as e:
            st.error(f"Error loading OGB arXiv dataset: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    elif dataset_name == "AMiner":
        # Load AMiner dataset with proper error handling
        try:
            import os
            import torch
            from torch_geometric.data import Data
            from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
            from torch_geometric.data.storage import GlobalStorage, EdgeStorage, NodeStorage

            # Add safe globals for AMiner dataset loading
            torch.serialization.add_safe_globals([
                DataEdgeAttr, DataTensorAttr, Data,
                GlobalStorage, EdgeStorage, NodeStorage
            ])

            aminer_processed_path = os.path.join('data', 'AMiner', 'processed')

            # Try loading from our custom homogeneous file (data_homogeneous.pt)
            homogeneous_file = os.path.join(aminer_processed_path, 'data_homogeneous.pt')
            if os.path.exists(homogeneous_file):
                try:
                    data = torch.load(homogeneous_file, map_location='cpu', weights_only=False)
                    st.success(f"Loaded AMiner dataset: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
                    return data
                except Exception as e:
                    st.warning(f"Could not load from {homogeneous_file}: {e}")

            # Fallback: Try loading from standard processed file (data.pt)
            standard_file = os.path.join(aminer_processed_path, 'data.pt')
            if os.path.exists(standard_file):
                try:
                    data = torch.load(standard_file, map_location='cpu', weights_only=False)

                    # Validate dataset has required attributes
                    if hasattr(data, 'x') and hasattr(data, 'edge_index'):
                         st.success(f"Loaded AMiner dataset: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
                         return data
                except Exception as e:
                    pass # Continue to next method

            # No valid dataset files found
            st.error("AMiner processed dataset not found or invalid.")
            st.info("To fix this:")
            st.info("1. Run the training notebook: `notebooks/GNN_AMiner.ipynb`")
            return None

        except Exception as e:
            st.error(f"Error loading AMiner dataset: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    return None

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def preprocess_text(text):
    """Clean and preprocess extracted text"""
    # Remove special characters and extra whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text

def text_to_features(text, target_dim=128):
    """Convert text to feature vector matching the model input"""

    # Validate text length before processing
    words = text.split()
    word_count = len(words)

    if word_count < 50:
        raise ValueError(f"Text too short ({word_count} words). Please provide at least 50 words for accurate feature extraction.")

    # Use TF-IDF to create a feature vector
    vectorizer = TfidfVectorizer(max_features=target_dim, stop_words='english')

    try:
        vector = vectorizer.fit_transform([text]).toarray()[0]

        # Check if vector is all zeros (no meaningful features extracted)
        if np.sum(vector) == 0:
            raise ValueError("Could not extract meaningful features from text. Please provide text with more varied vocabulary.")

        # Pad or truncate to exact target_dim
        if len(vector) < target_dim:
            vector = np.pad(vector, (0, target_dim - len(vector)))
        elif len(vector) > target_dim:
            vector = vector[:target_dim]

        # Normalize
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        return torch.FloatTensor(vector)

    except ValueError:
        # Re-raise our custom validation errors
        raise
    except Exception as e:
        raise ValueError(f"Feature extraction failed: {str(e)}. Please check your text content.")

def extract_real_subgraph(features, data, k_neighbors=10):
    """
    Extract real citation/collaboration network using K-nearest neighbors

    Args:
        features: Feature vector of the uploaded paper
        data: Real dataset with nodes and edges
        k_neighbors: Number of nearest neighbors to find

    Returns:
        subgraph_data: Data object with real citation network
        target_idx: Index of the uploaded paper in the subgraph
        node_mapping: Mapping from subgraph indices to original dataset indices
    """
    if data is None or not hasattr(data, 'num_nodes') or not hasattr(data, 'x'):
        raise ValueError("Dataset not available. Cannot create real citation network.")

    # Compute cosine similarity between uploaded paper and all papers in dataset
    features_norm = F.normalize(features.unsqueeze(0), p=2, dim=1)
    dataset_features_norm = F.normalize(data.x, p=2, dim=1)
    similarities = torch.mm(features_norm, dataset_features_norm.t()).squeeze()

    # Find K nearest neighbors
    top_k_values, top_k_indices = torch.topk(similarities, k=min(k_neighbors, data.num_nodes))
    nearest_neighbors = top_k_indices.tolist()

    # Extract 2-hop neighborhood from real citation/collaboration network
    subgraph_nodes = set(nearest_neighbors)

    # Add 1-hop neighbors
    edge_index = data.edge_index
    for node in nearest_neighbors:
        # Outgoing edges
        outgoing = edge_index[1, edge_index[0] == node].tolist()
        subgraph_nodes.update(outgoing[:5])  # Limit to 5 per node
        # Incoming edges
        incoming = edge_index[0, edge_index[1] == node].tolist()
        subgraph_nodes.update(incoming[:5])  # Limit to 5 per node

    # Add 2-hop neighbors (sample to keep graph manageable)
    first_hop = list(subgraph_nodes)
    for node in first_hop[:min(10, len(first_hop))]:  # Only from first 10 nodes
        outgoing = edge_index[1, edge_index[0] == node].tolist()
        subgraph_nodes.update(outgoing[:3])  # Add up to 3 second-hop neighbors

    # Limit total nodes for visualization
    subgraph_nodes = list(subgraph_nodes)[:100]
    subgraph_nodes_set = set(subgraph_nodes)

    # Create mapping from original indices to subgraph indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph_nodes)}

    # Extract edges that connect nodes in the subgraph
    mask = torch.tensor([
        (edge_index[0, i].item() in subgraph_nodes_set and
         edge_index[1, i].item() in subgraph_nodes_set)
        for i in range(edge_index.shape[1])
    ])

    subgraph_edges = edge_index[:, mask]

    # Remap edge indices to subgraph
    subgraph_edge_index = torch.LongTensor([
        [old_to_new[subgraph_edges[0, i].item()] for i in range(subgraph_edges.shape[1])],
        [old_to_new[subgraph_edges[1, i].item()] for i in range(subgraph_edges.shape[1])]
    ])

    # Get features for subgraph nodes
    subgraph_features = data.x[subgraph_nodes]

    # Add the uploaded paper as a new node
    all_features = torch.cat([subgraph_features, features.unsqueeze(0)], dim=0)
    target_idx = len(subgraph_nodes)

    # Connect uploaded paper to its K nearest neighbors in the subgraph
    new_edges_src = [target_idx] * len(nearest_neighbors) + nearest_neighbors
    new_edges_dst = nearest_neighbors + [target_idx] * len(nearest_neighbors)
    new_edges_src = [target_idx if old_idx not in old_to_new else old_to_new[old_idx]
                     for old_idx in new_edges_src if old_idx == target_idx or old_idx in old_to_new]
    new_edges_dst = [target_idx if old_idx not in old_to_new else old_to_new[old_idx]
                     for old_idx in new_edges_dst if old_idx == target_idx or old_idx in old_to_new]

    # Properly align the lists
    new_edge_pairs = []
    for i, neighbor in enumerate(nearest_neighbors):
        if neighbor in old_to_new:
            new_edge_pairs.append([target_idx, old_to_new[neighbor]])
            new_edge_pairs.append([old_to_new[neighbor], target_idx])

    if new_edge_pairs:
        new_edge_index = torch.LongTensor(new_edge_pairs).t()
        final_edge_index = torch.cat([subgraph_edge_index, new_edge_index], dim=1)
    else:
        final_edge_index = subgraph_edge_index

    # Create node mapping (subgraph index -> original dataset index)
    node_mapping = {i: subgraph_nodes[i] for i in range(len(subgraph_nodes))}
    node_mapping[target_idx] = -1  # -1 indicates the uploaded paper

    # Create reverse mapping for K-nearest neighbors with their similarity scores
    knn_similarity_map = {}
    for i, neighbor_idx in enumerate(nearest_neighbors):
        if neighbor_idx in old_to_new:
            subgraph_idx = old_to_new[neighbor_idx]
            knn_similarity_map[subgraph_idx] = top_k_values[i]

    subgraph_data = Data(x=all_features, edge_index=final_edge_index)

    return subgraph_data, target_idx, node_mapping, knn_similarity_map

def predict_topic(text, model, model_name, data, dataset_config):
    """Predict paper topic using the selected model"""
    # Convert text to features
    features = text_to_features(text, target_dim=dataset_config['feature_dim'])

    # Extract real citation subgraph with K-nearest neighbors
    graph_data, target_idx, node_mapping, knn_similarity_map = extract_real_subgraph(features, data, k_neighbors=10)

    # Predict
    with torch.no_grad():
        output = model(graph_data.x, graph_data.edge_index)
        probabilities = F.softmax(output[target_idx], dim=0)
        predicted_class = probabilities.argmax().item()
        confidence = probabilities[predicted_class].item()

    # Get top 5 predictions
    top5_probs, top5_indices = torch.topk(probabilities, k=min(5, len(probabilities)))

    return predicted_class, confidence, probabilities, top5_probs, top5_indices, graph_data, target_idx, node_mapping, knn_similarity_map

def create_knowledge_graph_visualization(graph_data, target_idx, categories, predicted_class, top5_indices, top5_probs, confidence, node_mapping, knn_similarity_map, dataset_name):
    """Create interactive knowledge graph visualization using Plotly with real citation network"""

    # Limit to small subgraph for visualization
    num_nodes = min(50, graph_data.x.shape[0])

    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes
    for i in range(num_nodes):
        node_type = "target" if i == target_idx else "context"
        G.add_node(i, node_type=node_type)

    # Add edges (limit to nodes in our subset) - REAL edges from dataset
    edge_index = graph_data.edge_index.numpy()
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < num_nodes and dst < num_nodes:
            G.add_edge(int(src), int(dst))

    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create edge trace
    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    # Create node traces with REAL paper IDs
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    node_sizes = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        if node == target_idx:
            # Uploaded paper
            node_colors.append('red')
            node_sizes.append(30)
            category_name = categories[predicted_class]
            node_text.append(f"Your Paper<br>Predicted: {category_name}<br>Confidence: {confidence:.1%}")
        elif node in node_mapping:
            original_idx = node_mapping[node]
            if original_idx != -1:
                # Real paper from dataset
                if node in knn_similarity_map:
                    # K-nearest neighbor with actual similarity score
                    node_colors.append('orange')
                    node_sizes.append(20)
                    sim_score = knn_similarity_map[node]
                    if dataset_name == "OGB arXiv":
                        node_text.append(f"arXiv Paper #{original_idx}<br>Similarity: {sim_score:.3f}<br>Citation Connection")
                    else:
                        node_text.append(f"Author #{original_idx}<br>Similarity: {sim_score:.3f}<br>Collaboration Connection")
                else:
                    # Citation/collaboration neighbor
                    node_colors.append('lightblue')
                    node_sizes.append(12)
                    if dataset_name == "OGB arXiv":
                        node_text.append(f"arXiv Paper #{original_idx}<br>Connected via citations")
                    else:
                        node_text.append(f"Author #{original_idx}<br>Connected via co-authorship")
            else:
                # Uploaded paper (shouldn't happen but handle gracefully)
                node_colors.append('red')
                node_sizes.append(30)
                node_text.append(f"Your Paper")
        else:
            # Fallback
            node_colors.append('lightgray')
            node_sizes.append(8)
            node_text.append(f"Node {node}")

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            showscale=False,
            color=node_colors,
            size=node_sizes,
            line=dict(width=1, color='#333')
        ),
        showlegend=False
    )

    # Create figure
    network_type = "Citation Network" if dataset_name == "OGB arXiv" else "Collaboration Network"
    fig = go.Figure(data=[edge_trace, node_trace],
                   # Layout configuration
                   layout=go.Layout(
                       title=dict(
                           text=f'Real {network_type} (K-NN Subgraph)',
                           x=0.5,
                           xanchor='center'
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20, l=5, r=5, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       plot_bgcolor='white',
                       height=500
                   ))

    return fig

def create_topic_distribution_graph(probabilities, categories, top5_indices):
    """Create an interactive topic distribution visualization"""

    # Get all probabilities
    all_probs = probabilities.numpy()
    top_k = min(10, len(all_probs))
    top_indices = np.argsort(all_probs)[-top_k:][::-1]

    fig = go.Figure(data=[
        go.Bar(
            x=[categories[i] for i in top_indices],
            y=[all_probs[i] * 100 for i in top_indices],
            marker=dict(color='steelblue'),
            text=[f"{all_probs[i]:.1%}" for i in top_indices],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Topic Probability Distribution',
        xaxis_title='Research Topics',
        yaxis_title='Confidence (%)',
        xaxis={'tickangle': -45},
        height=400,
        showlegend=False
    )

    return fig

def main():
    # Header
    st.markdown('<div class="main-header">Research Compass</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">GNN-based Research Paper Classification</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("Research Compass")
        
        st.subheader("Configuration")
        
        # Dataset selection
        dataset_choice = st.selectbox(
            "Select Dataset",
            ["OGB arXiv", "AMiner"],
            help="Choose the dataset for prediction"
        )
        
        # Set default based on dataset
        if dataset_choice == "AMiner":
            default_idx = 2  # GraphSAGE (strongly recommended)
        else:
            default_idx = 2  # GraphSAGE (slight edge, consistency)

        model_choice = st.selectbox(
            "Select GNN Model",
            ["GAT (Graph Attention)", "GCN (Graph Convolution)", "GraphSAGE"],
            index=default_idx,
            help="Choose the Graph Neural Network architecture"
        )
        
        st.markdown("---")
        st.markdown("### Training Notebooks")
        st.markdown("""
        View complete training code:
        - [OGB Training](https://github.com/Apc0015/Research_Compass_GNN/blob/main/notebooks/GNN_OGB.ipynb)
        - [AMiner Training](https://github.com/Apc0015/Research_Compass_GNN/blob/main/notebooks/GNN_AMiner.ipynb)

        [**View Project Repository**](https://github.com/Apc0015/Research_Compass_GNN)

        Retrain models by running all cells in these notebooks.
        """)
    
    # Load models
    with st.spinner(f"Loading {dataset_choice} models..."):
        gat, gcn, sage, results, dataset_config = load_models(dataset_choice)
        data = load_dataset_for_features(dataset_choice)
    
    if gat is None:
        st.error(f"Models not found for {dataset_choice}. Please train the models first by running the notebook.")
        if dataset_choice == "OGB arXiv":
            st.info("Run all cells in `notebooks/GNN_OGB.ipynb` to train and save the models.")
        else:
            st.info("Run all cells in `notebooks/GNN_AMiner.ipynb` to train and save the models.")
        return
        
    if data is None:
        st.error("Dataset not loaded. Real citation/collaboration network visualization requires the actual dataset.")
        st.info(f"Please run the training notebook to download and process the {dataset_choice} dataset.")
        return
    
    # Map model choice
    model_map = {
        "GAT (Graph Attention)": gat,
        "GCN (Graph Convolution)": gcn,
        "GraphSAGE": sage
    }
    selected_model = model_map[model_choice]
    
    # Get category mapping for current dataset
    categories = dataset_config['categories']
    
    # Main content
    # Main content
    tab_pred, tab_model, tab_data = st.tabs(["🔮 Prediction", "🧠 Model Architecture", "📊 Dataset Stats"])
    
    with tab_pred:
        st.markdown("### Upload Research Paper")
            
        # File uploader (multiple files)
        uploaded_files = st.file_uploader(
            "Choose PDF file(s)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more research papers in PDF format"
        )
        
        # Text input alternative
        with st.expander("Or paste paper abstract/text"):
            placeholder_text = "Enter the abstract or full text of your research paper..."
            if dataset_choice == "AMiner":
                placeholder_text = "Enter author information or research topics..."
            manual_text = st.text_area(
                "Paste text here",
                height=200,
                placeholder=placeholder_text
            )
        
        # Prediction button
        if st.button("Predict Topic", type="primary", use_container_width=True):
            papers_to_process = []
            
            # Get text from files or manual input
            if uploaded_files and len(uploaded_files) > 0:
                with st.spinner(f"Extracting text from {len(uploaded_files)} PDF(s)..."):
                    for uploaded_file in uploaded_files:
                        paper_text = extract_text_from_pdf(uploaded_file)
                        if paper_text:
                            paper_text = preprocess_text(paper_text)
                            papers_to_process.append((uploaded_file.name, paper_text))
                    st.success(f"Extracted text from {len(papers_to_process)} file(s)")
            elif manual_text:
                paper_text = preprocess_text(manual_text)
                papers_to_process.append(("Manual Input", paper_text))
            else:
                st.warning("Please upload a PDF or paste text to continue.")
                st.stop()
            
            # Process each paper
            for paper_idx, (paper_name, paper_text) in enumerate(papers_to_process):
                if not paper_text:
                    st.error(f"No text extracted from '{paper_name}'")
                    continue

                # Validate text length
                word_count = len(paper_text.split())

                # Show text statistics
                st.markdown(f"#### Processing: {paper_name}")
                st.info(f"Text length: {word_count} words")

                # Check minimum word count
                if word_count < 50:
                    st.error(f"Text too short: {word_count} words. Need at least 50 words for accurate prediction.")
                    st.info("Requirements:")
                    st.info("- Minimum: 50 words")
                    st.info("- Recommended: 200+ words (typical abstract length)")
                    st.info("- Ideal: 500+ words (abstract + introduction)")
                    continue
                elif word_count < 200:
                    st.warning(f"Short text: {word_count} words. Predictions may be less accurate. Recommend 200+ words.")

                # Make prediction
                with st.spinner(f"Analyzing '{paper_name}' with {model_choice} on {dataset_choice}..."):
                    try:
                        predicted_class, confidence, all_probs, top5_probs, top5_indices, graph_data, target_idx, node_mapping, knn_similarity_map = predict_topic(
                            paper_text, selected_model, model_choice, data, dataset_config
                        )
                    except ValueError as e:
                        st.error(f"Validation error: {e}")
                        continue
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                        continue
                    
                    # Display results
                    st.markdown("---")
                    if len(papers_to_process) > 1:
                        st.markdown(f"## Prediction Results for '{paper_name}'")
                    else:
                        st.markdown("## Prediction Results")
                    
                    # Main prediction
                    st.markdown(f"""
                    <div style="
                        background-color: #f8f9fa;
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 5px solid #4CAF50;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin-bottom: 20px;
                    ">
                        <h3 style="margin:0; color: #666; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Predicted Topic</h3>
                        <h1 style="margin: 10px 0; color: #333; font-size: 32px;">{categories[predicted_class]}</h1>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 15px;">
                            <span style="font-weight: bold; color: #4CAF50; font-size: 18px;">{confidence:.1%} Confidence</span>
                            <span style="color: #888; font-size: 12px;">{dataset_choice} • {model_choice}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Explainability (Idea 7)
                    with st.expander("🤔 Why this prediction?"):
                        st.markdown(f"""
                        **Confidence Analysis:**
                        The model is **{confidence:.1%}** confident in this prediction.
                        
                        **Graph Context:**
                        The prediction is based on the paper's text features and its connections in the citation graph.
                        - **Direct Neighbors:** The paper is connected to {len(top5_indices)} similar papers in the graph.
                        - **Dominant Category:** The majority of neighbors likely belong to **{categories[predicted_class]}**.
                        """)

                    # Knowledge Graph Visualization
                    st.markdown("---")
                    st.markdown("### Knowledge Graph Visualization")
                    
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        # Citation network graph
                        kg_fig = create_knowledge_graph_visualization(
                            graph_data, target_idx, categories, predicted_class, top5_indices, top5_probs, confidence, node_mapping, knn_similarity_map, dataset_choice
                        )
                        st.plotly_chart(kg_fig, use_container_width=True)
                    
                    with col_viz2:
                        # Topic distribution
                        dist_fig = create_topic_distribution_graph(all_probs, categories, top5_indices)
                        st.plotly_chart(dist_fig, use_container_width=True)
                    
                    # Top 5 predictions
                    st.markdown("---")
                    st.markdown("### Top 5 Predictions")
                    
                    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
                        idx_val = idx.item()
                        prob_val = prob.item()
                        
                        col_a, col_b = st.columns([3, 1])
                        with col_a:
                            st.markdown(f"**{i+1}. {categories[idx_val]}**")
                            st.progress(prob_val)
                        with col_b:
                            st.metric("", f"{prob_val:.1%}")
                    
                    # Show paper preview
                    st.markdown("---")
                    st.markdown("### Paper Preview")
                    with st.expander("View extracted text (first 1000 characters)"):
                        st.text(paper_text[:1000] + "..." if len(paper_text) > 1000 else paper_text)
                    
                    # Download predictions
                    st.markdown("---")
                    predictions_df = pd.DataFrame({
                        'Rank': range(1, 6),
                        'Topic': [categories[idx.item()] for idx in top5_indices],
                        'Confidence': [f"{prob.item():.2%}" for prob in top5_probs]
                    })
                    
                    st.download_button(
                        label="Download Predictions (CSV)",
                        data=predictions_df.to_csv(index=False),
                        file_name=f"predictions_{paper_name.replace('.pdf', '')}_{dataset_choice.lower().replace(' ', '_')}.csv",
                        mime="text/csv",
                        key=f"download_btn_{paper_idx}_{paper_name}"
                    )

        # Footer
        st.markdown("---")
        st.markdown(f"""
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
            <p>Dataset: {dataset_choice} | Model: {model_choice}</p>
        </div>
        """, unsafe_allow_html=True)

    with tab_model:
        st.markdown("### Model Architecture")
        if model_choice == "GAT (Graph Attention)":
            st.markdown("""
            **Graph Attention Network (GAT)**
            - **Mechanism**: Uses attention layers to weigh the importance of different neighbors.
            - **Best for**: Citation networks where some references are more relevant than others.
            - **Layers**: 3 (OGB) or 2 (AMiner)
            - **Heads**: 4 attention heads per layer
            """)
        elif model_choice == "GCN (Graph Convolution)":
            st.markdown("""
            **Graph Convolutional Network (GCN)**
            - **Mechanism**: Aggregates neighbor features using a fixed normalization.
            - **Best for**: Homophilous graphs (neighbors are similar).
            - **Layers**: 3 (OGB) or 2 (AMiner)
            """)
        else:
            st.markdown("""
            **GraphSAGE**
            - **Mechanism**: Samples and aggregates neighbors (Inductive learning).
            - **Best for**: Large graphs and unseen nodes (like new papers).
            - **Layers**: 3 (OGB) or 2 (AMiner)
            """)

    with tab_data:
        st.markdown(f"### {dataset_choice} Statistics")
        st.markdown(f"""
        - **Description**: {dataset_config['description']}
        - **Task**: {dataset_config['task']}
        - **Classes**: {dataset_config['num_classes']}
        - **Feature Dimension**: {dataset_config['feature_dim']}
        """)
        if dataset_choice == "OGB arXiv":
            st.info("The OGB arXiv dataset represents a citation network of Computer Science papers.")
        else:
            st.info("The AMiner dataset represents a co-authorship network of researchers.")

if __name__ == "__main__":
    main()
