"""Model builders for text and image classifiers."""

import timm
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification

from config import LABELS


def build_text_model(model_name, device):
    """Build a Hugging Face transformer for multi-label classification."""
    cfg = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)
    return model.to(device), cfg


def build_image_model(model_name, device):
    """Build a timm image model for multi-label classification."""
    model = timm.create_model(model_name, pretrained=True, num_classes=len(LABELS))
    return model.to(device)


def compute_pos_weight(Y):
    """
    Compute positive class weights for BCEWithLogitsLoss.

    Helps handle class imbalance by giving more weight to rare genres.
    """
    pos = Y.sum(axis=0)
    neg = len(Y) - pos
    eps = 1e-6
    weights = (neg + eps) / (pos + eps)
    return torch.tensor(weights, dtype=torch.float32)
