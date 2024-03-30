# Transformer based model to play the Hangman game.

This document details the machine learning model I developed to play Hangman. My model leverages a Transformer based architecture to predict missing letters in masked words at each step of the hangman game. Transformers are known for its effectiveness in handling sequential data, making them particularly suitable for this task.

## 1. Overall Strategy

The core strategy involves training a model on a dataset of words and their letter distributions. Given a word with some letters masked (represented by '#'), the model predicts the most likely letters to fill in the blanks. These predictions guide letter choices during a Hangman game, aiming to complete the word before running out of attempts.

## 2. Code Breakdown

These are the order of the components in my network architecture:

1.	Token embedding
2.	Positional Encoding
3.	A sequence of encoder blocks. Each encoder block has a layer normalization, multiheaded-self attention block, skip-connection, layer normalization and a fully-connected layer with gelu activation function, skip-connection and a layer normalization.
4.	A classification head made of fully connected layers. 




The code is divided into three files: data_loader.py, transformer.py and network_comps.py.

### File 1: network_comps.py

This file contains the following components:

Positional Encoding: Injects positional information into letter (including mask) embeddings using sine and cosine functions. This helps the model understand the relative order of letters within a word, crucial for Hangman.

Encoder Layer: This is the core processing unit. It utilizes a multi-headed attention mechanism to analyze the relationships between masked and revealed letters in a word. Each layer refines the word representation, enhancing prediction accuracy.

Classification Layer: Takes the final encoded representation and predicts the probability of each letter appearing in any of the masked positions. These probabilities guide the model's guesses during Hangman.

### File 2: transformer.py

This file defines the overall model architecture:

Embedding Layer: Converts each letter (including '#') into a dense vector representation. Embedding dimension of 64 is used.
Positional Encoding: Applies Positional Encoding to the embedded word sequence.
Encoder Layers: Stacks 16 EncoderLayer instances with 4 heads for progressive refinement.
Classification Layer: Predicts the probability distribution for each masked letter position.

Log softmax is applied to the logits as KL divergence loss is used.

### File 3: data_loader.py

This file handles data preparation for training:

Tokenize Function: Converts words (including the hidden letters represented by #) into sequences of numerical indices for the model.

WordDataset Class:
Loads words from a text file.
Filters out words exceeding 16 characters or containing only one unique letter.
Randomly selects a subset of unique letters to mask (using #)
Generates labels representing the probability distribution of letters for masked positions.


## Training

In the train.ipynb jupyter notebook, I trained the model with a learning rate of 0.001 for 500 epochs using Adam optimizer. I used KL divergence loss as the loss function as I am trying to match the predicted distribution to the label distribution. The loss decreases from ~2 at epoch 0 to around ~1.23 at epoch 500. The final model is loaded into the hangman code provided to me. Since I trained the model only for words with 16 or fewer characters, when the input word is longer than 16 characters, I either choose the first or the last 16 characters to be the input word for the model. My model results in ~60% accuracy. This can possibly be improved further by fine tuning the model hyper-parameters such as the embedding dimension, number of heads and number of encoder layers etc.
