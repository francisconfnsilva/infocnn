import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import CONFIG

class CNN(nn.Module):
    def __init__(self, embedding_dim, dropout=CONFIG["dropout"]):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64 * 7 * 7, embedding_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1) # flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ContrastiveModel(nn.Module):
    def __init__(self, embedding_dim, num_classes=10):
        super().__init__()
        self.image_encoder = CNN(embedding_dim)
        self.label_encoder = nn.Embedding(num_classes, embedding_dim)

    def forward(self, images, labels):
        # encode images
        img_emb = self.image_encoder(images)
        
        # encode labels
        lbl_emb = self.label_encoder(labels)
        
        img_emb = F.normalize(img_emb, p=2, dim=1)
        lbl_emb = F.normalize(lbl_emb, p=2, dim=1)
        
        return img_emb, lbl_emb