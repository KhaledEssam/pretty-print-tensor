# PPT Pretty Print Tensor

PPT is an attempt to make debugging tensors easier. It basically assigns a unique prime value to each variable you define (unless you explicitly assign a specific value).

It assigns unique prime values so that if a layer has an output that is a multiple of a vector dimension, it can check which dimension divides it.

## Interface

PPT defines the class `PPT`, which is the main class that holds all the relevant information.

`PPT` has a `defvars` method which takes a list containing the names of the variables to be defined, and, optionally, a dict `values` if you want to manually set the values for specific values.

If you want to specify your own values, then you need to make sure the values are not duplicated in order for `PPT` to print the correct names.

## Usage

Calling a `PPT` instance on one or more `nn.Module`s returns the same modules with a `pp` attribute set to them. This attribute is then to be used in the `forward` method of the module.

```python
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, nhead: int = 1, **kwargs):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(2 * embedding_dim, nhead=nhead), num_layers=6
        )

    def forward(self, x: torch.Tensor, **kwargs):
        self.pp(x)
        embedded = self.embed(x)
        self.pp(embedded)
        embedded = embedded.transpose(0, 1)
        self.pp(embedded)
        embedded = torch.cat([embedded, embedded], dim=2)
        self.pp(embedded)
        encoded = self.encoder(embedded)
        self.pp(encoded)
        encoded = encoded.transpose(0, 1)
        self.pp(encoded, embedded)
        return encoded
```

And then in the `main.py`:


```python
import torch

from model import Model
from ppt import PPT


def main():
    pp = PPT()
    D = pp.defvars(
        ["vocab_size", "embedding_dim", "nhead", "seq_len", "batch_size"],
        values={"nhead": 2, "embedding_dim": 8},
    )
    model = pp(Model(**D))
    x = torch.randint(0, D["vocab_size"], (D["batch_size"], D["seq_len"]))
    pp(x)
    x = model(x)


if __name__ == "__main__":
    main()
```

Output

```bash
Defined Variables:
+---------------+-------+
| Variable Name | Value |
+---------------+-------+
|   vocab_size  |   2   |
| embedding_dim |   8   |
|     nhead     |   2   |
|    seq_len    |   3   |
|   batch_size  |   5   |
+---------------+-------+
+------------+---------------+------------+-----------------------+
|   Caller   | Variable Name |    Type    |         Shape         |
+------------+---------------+------------+-----------------------+
| main.py:15 |       x       | LongTensor | [batch_size, seq_len] |
+------------+---------------+------------+-----------------------+
+-----------------------------------+---------------+------------+-----------------------+
|               Caller              | Variable Name |    Type    |         Shape         |
+-----------------------------------+---------------+------------+-----------------------+
| /home/khaled/code/ppt/model.py:14 |       x       | LongTensor | [batch_size, seq_len] |
+-----------------------------------+---------------+------------+-----------------------+
+-----------------------------------+---------------+-------------+--------------------------------------+
|               Caller              | Variable Name |     Type    |                Shape                 |
+-----------------------------------+---------------+-------------+--------------------------------------+
| /home/khaled/code/ppt/model.py:16 |    embedded   | FloatTensor | [batch_size, seq_len, embedding_dim] |
+-----------------------------------+---------------+-------------+--------------------------------------+
+-----------------------------------+---------------+-------------+--------------------------------------+
|               Caller              | Variable Name |     Type    |                Shape                 |
+-----------------------------------+---------------+-------------+--------------------------------------+
| /home/khaled/code/ppt/model.py:18 |    embedded   | FloatTensor | [seq_len, batch_size, embedding_dim] |
+-----------------------------------+---------------+-------------+--------------------------------------+
+-----------------------------------+---------------+-------------+------------------------------------------+
|               Caller              | Variable Name |     Type    |                  Shape                   |
+-----------------------------------+---------------+-------------+------------------------------------------+
| /home/khaled/code/ppt/model.py:20 |    embedded   | FloatTensor | [seq_len, batch_size, 2 × embedding_dim] |
+-----------------------------------+---------------+-------------+------------------------------------------+
+-----------------------------------+---------------+-------------+------------------------------------------+
|               Caller              | Variable Name |     Type    |                  Shape                   |
+-----------------------------------+---------------+-------------+------------------------------------------+
| /home/khaled/code/ppt/model.py:22 |    encoded    | FloatTensor | [seq_len, batch_size, 2 × embedding_dim] |
+-----------------------------------+---------------+-------------+------------------------------------------+
+-----------------------------------+---------------+-------------+------------------------------------------+
|               Caller              | Variable Name |     Type    |                  Shape                   |
+-----------------------------------+---------------+-------------+------------------------------------------+
| /home/khaled/code/ppt/model.py:24 |    encoded    | FloatTensor | [batch_size, seq_len, 2 × embedding_dim] |
| /home/khaled/code/ppt/model.py:24 |    embedded   | FloatTensor | [seq_len, batch_size, 2 × embedding_dim] |
+-----------------------------------+---------------+-------------+------------------------------------------+

```