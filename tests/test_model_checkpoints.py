"""
Unit tests for Research Compass GNN model checkpoints and mappings.
"""
from pathlib import Path
import torch
import pytest

ROOT = Path(__file__).parent.parent
SAVED_MODELS_DIR = ROOT / "saved_models"


def test_ogb_checkpoint_structure():
    ckpt_path = SAVED_MODELS_DIR / "OGB_models.pt"
    assert ckpt_path.exists(), "OGB_models.pt checkpoint file missing"

    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    required_keys = ["gat", "gcn", "sage", "results"]
    for key in required_keys:
        assert key in data, f"Missing key '{key}' in OGB_models.pt"

    results = data["results"]
    assert "node_classification" in results
    assert "link_prediction" in results


def test_aminer_checkpoint_structure():
    ckpt_path = SAVED_MODELS_DIR / "aminer_models.pt"
    assert ckpt_path.exists(), "aminer_models.pt checkpoint file missing"

    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    required_keys = ["num_features", "num_classes", "sage_node", "results_node"]
    for key in required_keys:
        assert key in data, f"Missing key '{key}' in aminer_models.pt"

    assert data["num_features"] == 136
    assert data["num_classes"] == 8


def test_arxiv_categories_mapping():
    from app import ARXIV_CATEGORIES
    assert len(ARXIV_CATEGORIES) == 40
    assert ARXIV_CATEGORIES[0] == "cs.AI - Artificial Intelligence"
    assert ARXIV_CATEGORIES[22] == "cs.LG - Machine Learning"
