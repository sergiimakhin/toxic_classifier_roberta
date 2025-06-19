import torch.nn as nn
from transformers import RobertaForSequenceClassification
from config import MODEL_NAME

def get_model():
    return RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=6)