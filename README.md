# InfoNCE vs. MSE: A Contrastive Learning Case Study

This repository explores the difference between contrastive learning (InfoNCE) and standard regression (MSE) when learning image representations.

Using a simple CNN on the MNIST dataset, we show how contrastive objectives create more robust latent spaces compared to pixel-matching objectives.

## Objective
To compare two loss functions for alignment:
1.  **InfoNCE:** Maximizes similarity between an image and its label embedding while pushing away incorrect labels.
2.  **MSE:** Minimizes the Euclidean distance between an image and its label embedding without negative sampling.

## Key Findings
* InfoNCE creates distinct, well-separated clusters for each digit.
* The contrastive model maintains higher accuracy when images are corrupted with gaussian noise.
* InfoNCE often converges faster.

## Getting Started

**Install Dependencies:**
   ```bash
   pip install -r requirements.txt