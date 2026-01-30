import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE

from src.config import CONFIG


def get_data(batch_size, data_path='../data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, test_loader

def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    
    # pre-compute all 10 label embeddings
    all_labels = torch.arange(10).to(CONFIG["device"])
    with torch.no_grad():
        label_prototypes = model.label_encoder(all_labels)
        label_prototypes = F.normalize(label_prototypes, p=2, dim=1) # Normalize
    
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(CONFIG["device"]), targets.to(CONFIG["device"])
            
            # encode images
            img_emb = model.image_encoder(images)
            img_emb = F.normalize(img_emb, p=2, dim=1)
            
            similarity = torch.matmul(img_emb, label_prototypes.T)
            
            # prediction is the index with highest dot product
            predictions = torch.argmax(similarity, dim=1)
            
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            
    return 100 * correct / total

# plots

def get_embeddings(model, loader):
        model.eval()
        embs, lbls = [], []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(CONFIG["device"])
                output = model.image_encoder(images)
                output = F.normalize(output, p=2, dim=1)
                embs.append(output.cpu().numpy())
                lbls.append(targets.numpy())
            return np.concatenate(embs), np.concatenate(lbls)

# Evaluation

def test_noise(model, loader, noise_levels=[0.0, 0.1, 0.3, 0.5, 0.8]):
    model.eval()
    accuracies = []
    
    for noise in noise_levels:
        correct = 0
        total = 0
        
        all_labels = torch.arange(10).to(CONFIG["device"])
        with torch.no_grad():
            prototypes = F.normalize(model.label_encoder(all_labels), dim=1)

        for images, targets in loader:
            images, targets = images.to(CONFIG["device"]), targets.to(CONFIG["device"])
            
            # --- NOISE ---
            # add gaussian noise
            images = images + torch.randn_like(images) * noise
            images = torch.clamp(images, 0, 1)
            
            with torch.no_grad():
                img_emb = F.normalize(model.image_encoder(images), dim=1)
                similarity = torch.matmul(img_emb, prototypes.T)
                predictions = torch.argmax(similarity, dim=1)
                
            correct += (predictions == targets).sum().item()
            total += targets.size(0)
            
        accuracies.append(100 * correct / total)
    
    return accuracies

def plot_negative_sample(model, loader):
    model.eval()
    images_batch, labels_batch = next(iter(loader))
    images_batch, labels_batch = images_batch.to(CONFIG["device"]), labels_batch.to(CONFIG["device"])
    
    with torch.no_grad():
        img_emb = F.normalize(model.image_encoder(images_batch), dim=1)
        all_classes = torch.arange(10).to(CONFIG["device"])
        prototypes = F.normalize(model.label_encoder(all_classes), dim=1)
        sim_matrix = torch.matmul(img_emb, prototypes.T)
        
    batch_indices = torch.arange(len(images_batch))
    correct_scores = sim_matrix[batch_indices, labels_batch]
    
    worst_idx = torch.argmin(correct_scores).item()

    prediction_scores = sim_matrix[worst_idx]
    predicted_label = torch.argmax(prediction_scores).item()
    actual_label = labels_batch[worst_idx].item()
    
    img = images_batch[worst_idx].cpu().squeeze()
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {actual_label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()