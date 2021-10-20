import torch
import os

DATA_PATH = "Food-5K"
BASE_PATH = "dataset"

TRAIN = os.path.join(BASE_PATH, "training")
VAL = os.path.join(BASE_PATH, "validation")
TEST = os.path.join(BASE_PATH, "evaluation")

# initialize the list of class label names
CLASSES = ["Bread", "Dairy_product", "Dessert", "Egg", "Fried_food",
    "Meat", "Noodles/Pasta", "Rice", "Seafood", "Soup", "Vegetable/Fruit"]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

DEVICE = torch.device("cuda")

LOCAL_BATCH_SIZE = 32
PRED_BATCH_SIZE = 4
EPOCHS = 10
LR = 0.0001

PLOT_PATH = os.path.join("output", "model_training.png")
MODEL_PATH = os.path.join("output", "food_classifier.pth")
