# Video Game Genre Classifier

A **multi-label** classification model that predicts video game genres from text metadata (title + description) and in-game screenshots, using transformer ensembles and stacking.


---

## Problem Statement

Given a video game's **title**, **description**, and **screenshot**, predict which of 16 genres it belongs to. Since a single game can have multiple genres (e.g., "Action" + "RPG" + "Multiplayer"), this is a **multi-label classification** problem.

**Genres:** Action, Adventure, Arcade, Board Games, Card, Educational, Family, Fighting, Indie, Massively Multiplayer, Platformer, Puzzle, Racing, Shooter, Sports, Strategy

**Evaluation Metric:** Macro-averaged F1 Score

---

## Tools

**Python, PyTorch, Hugging Face Transformers, timm, Scikit-learn, iterative-stratification, Optuna, Pandas, NumPy.**

---

## Approach

### Architecture Overview

![Alternative Text](/Architecture.png)
### 1. Text Models (DeBERTa-v3 + RoBERTa)
- Combined **title + description** into a single text input
- Fine-tuned **DeBERTa-v3-xsmall** and **RoBERTa-base** as diverse backbones
- Used **Asymmetric Focal Loss** to handle severe class imbalance (down-weights easy negatives)
- **5-fold MultilabelStratifiedKFold** for robust cross-validation
- **Cosine annealing LR scheduler** + **mixed precision (AMP)**

### 2. Image Model (ConvNeXt-Tiny)
- Fine-tuned **ConvNeXt-Tiny** (pretrained on ImageNet) on game screenshots
- Standard augmentation: RandomResizedCrop, HorizontalFlip
- **BCEWithLogitsLoss** with computed **pos_weight** for class imbalance

### 3. Ensemble Strategy
- **Weighted average**: 40% DeBERTa + 40% RoBERTa + 20% ConvNeXt
- **Stacking**: Per-class Logistic Regression meta-learner trained on OOF probabilities from all 3 models

---

## Results

| Model | Best Fold Val F1 | Notes |
|-------|-----------------|-------|
| DeBERTa-v3-xsmall | 0.62 | 4 epochs, ASL loss |
| RoBERTa-base | 0.76 | 3 epochs, ASL loss |
| ConvNeXt-Tiny | 0.30 | 1 epoch, pos_weight BCE |

---

## Project Structure

```
video-game-genre-classifier/
├── README.md
├── config.py             # All hyperparameters
├── dataset.py            # Text & Image datasets
├── losses.py             # Asymmetric Focal Loss
├── models.py             # Text & Image model builders
├── train.py              # Training pipeline
├── ensemble.py           # Weighted avg + stacking
├── requirements.txt
```

---


##  Possible Improvements

- Train image model for more epochs (currently only 1)
- Use larger text models (DeBERTa-v3-large)
- Add more image augmentations 


---

## Author

**ibrahim Alshaya**
