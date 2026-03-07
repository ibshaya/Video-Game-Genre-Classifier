"""Configuration for Video Game Genre Classifier."""

SEED = 42

# Paths
TRAIN_CSV = "data/train.csv"
TEST_CSV = "data/test.csv"
IMG_ROOT = "data/images"
OUT_DIR = "outputs"

#  Text Models 
TEXT_MODEL_1 = "microsoft/deberta-v3-xsmall"
TEXT_MODEL_2 = "roberta-base"
MAX_LEN = 256

# Image Model 
IMG_MODEL = "convnext_tiny"
IMG_SIZE = 224

# Training 
NFOLDS = 5
EPOCHS_TEXT = 4
EPOCHS_IMG = 1
BATCH_TRAIN = 8
BATCH_VALID = 8
LR_TEXT = 2e-5
LR_IMG = 1e-4
WEIGHT_DECAY = 2e-4

# Ensemble Weights 
W_TEXT1 = 0.4
W_TEXT2 = 0.4
W_IMG = 0.2

#  Genre Labels (16 classes) 
LABELS = [
    "Action", "Adventure", "Arcade", "Board Games", "Card",
    "Educational", "Family", "Fighting", "Indie",
    "Massively Multiplayer", "Platformer", "Puzzle",
    "Racing", "Shooter", "Sports", "Strategy",
]
