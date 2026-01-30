import torch
import torch.nn as nn
from src.models import ContrastiveModel
from src.loss import info_nce_loss
from src.config import CONFIG
import torch.optim as optim

def reset():
    model = ContrastiveModel(CONFIG["embedding_dim"]).to(CONFIG["device"])
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    return model, optimizer

def train_epoch(model, optimizer, loader, loss_type="infonce"):
    model.train()
    total_loss = 0
    
    mse_criterion = nn.MSELoss()
    
    for images, targets in loader:
        images, targets = images.to(CONFIG["device"]), targets.to(CONFIG["device"])
        optimizer.zero_grad()
        
        # forward
        img_emb, lbl_emb = model(images, targets)
        
        if loss_type == "infonce":
            loss = info_nce_loss(img_emb, lbl_emb, CONFIG["temperature"])
        elif loss_type == "mse":
            loss = mse_criterion(img_emb, lbl_emb)
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)