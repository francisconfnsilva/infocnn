import torch
import torch.nn.functional as F

def info_nce_loss(image_embeddings, label_embeddings, temperature=0.1):
   
    # compute similarity
    # shape: (batch_size, batch_size)
    logits = torch.matmul(image_embeddings, label_embeddings.T) / temperature
    
    # targets are the indices [0, 1, ..., batch_size-1]
    batch_size = image_embeddings.shape[0]
    labels = torch.arange(batch_size).to(image_embeddings.device)
    
    loss = F.cross_entropy(logits, labels)
    
    return loss