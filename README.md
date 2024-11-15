# Sparse Autoencoder Tutorial

This repository provides a guide for implementing and training a **sparse autoencoder** using PyTorch.

## Setup

1. Get repository

Clone the repository and navigate into it:

    git clone https://github.com/your-username/sparse_autoencoder_tutorial.git
    cd sparse_autoencoder_tutorial

2. Dependencies

Install with:

    pip install -r requirements.txt

3. Data generation

Generate sample data (or replace it with actual activations):

    python data/sample_data_generation.py

4. Training the Sparse Autoencoder

Run the training script:
    
    python -m src.train

This will save the trained model as sparse_autoencoder.pth.

Monitor the loss values printed to the console to see the effect of sparse regularization.

5. Jupyter Tests

Open the Jupyter notebook for tests:

    jupyter notebook notebooks/tests.ipynb

In the notebook, you can:
- Visualize encoded representations.
- Compare original and reconstructed data to see the effects of sparsity.

6. Experiment with it!
    
E.g. - Try chaing training epochs.
