import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

def tokenize(word: str , max_len: int) -> Tensor:
    assert len(word) <= (max_len - 2), f"word length [{len(word)}] is greater than max_len [{max_len-2}]"
    ltr2idx = {ltr: idx+1 for idx, ltr in enumerate('#abcdefghijklmnopqrstuvwxyz')}
    tokenized_word = [28] + [ltr2idx[ltr] for ltr in word] + [29] + [0]*(max_len - 2 - len(word))
    return tokenized_word

class WordDataset(Dataset):
    def __init__(self, word_file, max_len, device):
        words = [word for word in open(word_file).read().split('\n') if len(word) <= (max_len - 2) and len(list(set(word)))>1]
        selected_indices = np.random.choice(np.arange(len(words)), len(words), replace=False)
        self.words = [words[i] for i in selected_indices]
        self.max_len = max_len
        self.device = device
    
    
    
    def __len__(self):
        return len(self.words)
    
    def __getitem__(self, idx):
        word = self.words[idx]
        unique_letters = list(set(word))
        num_letters_to_mask = np.random.randint(1, len(unique_letters))
        letters_to_mask = np.random.choice(unique_letters, num_letters_to_mask, replace=False)
        masked_word = ''.join(['#' if ltr in letters_to_mask else ltr for ltr in word])
        tokenized_word = torch.tensor(tokenize(masked_word, self.max_len)).to(self.device)
        onehot = np.zeros(26)
        for letter in letters_to_mask:
            onehot[ord(letter) - ord('a')] += word.count(letter)
        sumprobs = np.sum(onehot)
        onehot_label = torch.tensor([prob/sumprobs if prob > 0 else 1e-10 for prob in onehot]).to(self.device)
        return tokenized_word, onehot_label