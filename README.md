# PCA, Kernel PCA, Autoencoders & Variational Autoencoders

This repository contains the implementation of **Exercise 5** for the Computer Vision course at Computer Engineering and Informatics Department at University of Patras, focusing on dimensionality reduction and generative models applied to the **MNIST** dataset. The work covers classical linear methods (PCA), non-linear extensions (kernel PCA), and deep-learning-based methods (autoencoders and variational autoencoders).

The work begins with the mathematical formulation of PCA as a constrained optimization problem and its connection to the eigen-decomposition/SVD of the covariance matrix, including the low-rank approximation result that justifies truncation to the first LL**L** components.

On the practical side, PCA is implemented via SVD on selected MNIST digits to compute mean images, covariance matrices, principal components and reconstructions for varying numbers of components, along with reconstruction-error histograms and MSE curves. The same analysis is extended with **kernel PCA** using a radial basis kernel to obtain non-linear embeddings and visualize the digits in 2D/3D feature space, again comparing reconstruction quality as the number of kernel components increases.

Subsequently, several autoencoder architectures are trained on MNIST: a **linear undercomplete AE** whose learned subspace is quantitatively compared to PCA (via SVD of the decoder weights, cosine similarity and MSE), and deeper **non-linear AEs** with different parameter budgets and encoder–decoder tying strategies (including pseudo-inverse based decoders) to explore the effect of architecture and capacity on reconstruction error. Finally, a **Variational Autoencoder** with Gaussian latent variables is implemented and interpreted through the lens of probabilistic PCA, with experiments demonstrating the trade-off between reconstruction fidelity and latent-space regularization via sampling and generative examples from the learned latent space.

---

## Overview

The main objectives of this exercise are:

- To formulate and understand **Principal Component Analysis (PCA)** as a constrained optimization problem and its connection to eigenvalue decomposition / SVD of the covariance matrix.
- To apply **PCA** and **kernel PCA** for visualization and reconstruction of handwritten digits.
- To implement and compare **linear** and **non-linear autoencoders (AEs)** as learned dimensionality-reduction models.
- To implement a **Variational Autoencoder (VAE)** and interpret it in relation to probabilistic PCA and generative modeling.

---

## Implemented Methods

### 1. Principal Component Analysis (PCA)

- Computation of:
  - Mean image and covariance matrix for selected MNIST digits.
  - Principal components via SVD.
- Image reconstruction using the first \(L\) principal components for different values of \(L\).
- Evaluation of reconstruction quality with:
  - Mean Squared Error (MSE) per number of components.
  - Histograms / distributions of reconstruction errors.

### 2. Kernel PCA

- Implementation of **kernel PCA** using a radial basis function (RBF) kernel.
- Non-linear embeddings of MNIST digits in:
  - 2D and 3D feature spaces for visualization.
- Reconstruction from kernel PCA components and comparison of reconstruction error curves with linear PCA.

### 3. Autoencoders (AEs)

- **Linear undercomplete autoencoder**:

  - Trained on MNIST as a learned linear subspace model.
  - Quantitative comparison with PCA:
    - SVD of decoder weights.
    - Cosine similarity between bases.
    - MSE comparison.
- **Non-linear autoencoders**:

  - Deeper architectures with different parameter budgets.
  - Experiments with:
    - Tied vs. untied encoder–decoder weights.
    - Pseudo-inverse–based decoders.
  - Analysis of how architecture and capacity influence reconstruction performance.

### 4. Variational Autoencoder (VAE)

- Implementation of a **VAE** with Gaussian latent variables.
- Interpretation as a probabilistic latent variable model related to probabilistic PCA.
- Experiments demonstrating:
  - Trade-off between reconstruction accuracy and latent space regularization.
  - Sampling from the latent space to generate new digit images.
  - Latent-space structure visualizations (e.g. 2D latent grids, interpolations).

---

## Files

- `PCA.ipynb`: Classical PCA on MNIST (subspace, reconstruction, error analysis).
- `kernelPCA.ipynb`: Kernel PCA with RBF kernel and non-linear embeddings.
- `Autoencoders.ipynb`: Linear and non-linear autoencoder experiments.
- `VAE.ipynb`: Variational Autoencoder training and generative experiments.
- `Άσκηση5_1084674.pdf`, `CV_5_AUTOENCODERS.pdf`: Exercise description and theoretical background (in Greek and/or English).

---

## Requirements & Usage

- Python 3.x
- Typical scientific Python stack (e.g. `numpy`, `matplotlib`, `scikit-learn`)
- Deep learning framework (e.g. `PyTorch` or `TensorFlow`, depending on implementation)
- Jupyter Notebook

To run the experiments:

1. Create and activate a Python environment with the required packages.
2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
