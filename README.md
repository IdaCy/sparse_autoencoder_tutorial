# Sparse Autoencoder Tutorial

This project teaches the basics of sparse autoencoders and how to apply them to transformer neuron activations.

## Setup

1. Clone the repository and navigate into it:

    git clone https://github.com/your-username/sparse_autoencoder_tutorial.git
    cd sparse_autoencoder_tutorial

2. Install dependencies:

    pip install -r requirements.txt

3. Generate sample data (or replace it with actual activations):

    python data/sample_data_generation.py

4. Training the Sparse Autoencoder

Run the training script:
    
    python -m src.train

This will save the trained model as sparse_autoencoder.pth.

Monitor the loss values printed to the console to see the effect of sparse regularization.

5. Jupyter Tests

Open the Jupyter notebook for tests:

    jupyter notebook notebooks/tests.ipynb

6. Experiment with it!
    
    E.g. - Try chaing training epochs.

In the notebook, you can:
- Visualize encoded representations.
- Compare original and reconstructed data to see the effects of sparsity.
