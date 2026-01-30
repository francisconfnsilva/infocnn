import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay":1e-4,
    "dropout": 0.2,
    "epochs": 15,
    "embedding_dim": 64,    # Dimension of the shared vector space
    "temperature": 0.07,    # Softmax temperature for InfoNCE
    "device": device,
    "seed": 42           
}