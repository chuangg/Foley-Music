from torch import Tensor
import torch


def compute_epiano_accuracy(out: Tensor, tgt: Tensor, pad_idx=240) -> Tensor:
    """
    ----------
    Author: Damon Gwinn
    ----------
    Computes the average accuracy for the given input and output batches. Accuracy uses softmax
    of the output.
    ----------
    """

    # softmax = nn.Softmax(dim=-1)
    out = torch.argmax(out, dim=-1)

    out = out.flatten()
    tgt = tgt.flatten()

    mask = (tgt != pad_idx)

    out = out[mask]
    tgt = tgt[mask]

    # Empty
    if (len(tgt) == 0):
        return torch.tensor(1.0)

    num_right = (out == tgt)
    num_right = torch.sum(num_right).float()

    acc = num_right / len(tgt)

    return acc
