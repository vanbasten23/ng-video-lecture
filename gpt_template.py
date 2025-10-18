# This file is a template for gpt.py. Don't edit it directly.
# Instead, make a copy and edit the copy.
# It is used to practice implementing GPT from scratch.
# To use, make a copy of it and fill out the missing parts.
# Practice date:
# Time taken:
# Suggested implmentation order:
# - set random seed.
# - training script.
# - generate from the model.
# - get_batch function.
# - GPTLanguageModel class and its methods.


import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

# TODO: set random seed


# If you don't have the input.txt file, download it via "wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt".
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):  # split is either 'train' or 'val'.
    # generate a small batch of data of inputs x and targets y
    # TODO: implement this function
    pass
  
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        # TODO: implement this function
        pass

    def forward(self, x):
        # TODO: implement this function
        pass
      

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        # TODO: implement this function
        pass

    def forward(self, x):
        # TODO: implement this function
        pass

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        # TODO: implement this function
        pass

    def forward(self, x):
        # TODO: implement this function
        pass

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # TODO: implement this function
        pass

    def forward(self, x):
        # TODO: implement this function
        pass


class GPTLanguageModel(nn.Module):

    def __init__(self):
        # TODO: implement this function
        pass

    def _init_weights(self, module):
        # TODO: implement this function
        pass

    def forward(self, idx, targets=None):
        # TODO: implement this function
        pass

    def generate(self, idx, max_new_tokens):
        # TODO: implement this function
        pass

model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# TODO: create a PyTorch optimizer
#  optimizer = 

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # TODO: sample a batch of data
    # xb, yb =

    # TODO: evaluate the loss


# TODO: generate from the model


