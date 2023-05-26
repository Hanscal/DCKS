import os
import torch
import random
import numpy as np
from config import args


def set_seed():
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def save_config():
    if not args.test:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        with open(args.save_path + "/config.txt", "w") as the_file:
            for k, v in args.args.__dict__.items():
                if "False" in str(v):
                    continue
                elif "True" in str(v):
                    the_file.write("--{} ".format(k))
                else:
                    the_file.write("--{} {} ".format(k, v))

def print_opts(opts):
    """Prints the values of all command-line arguments."""
    print("=" * 80)
    print("Opts".center(80))
    print("-" * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print("{:>30}: {:<30}".format(key, str(opts.__dict__[key])).center(80))
    print("=" * 80)


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': f'{total_num:,}', 'Trainable': f'{trainable_num:,}'}


def embedding_similarity(
    batch,
    similarity: str = "cosine",
    reduction: str = "none",
    zero_diagonal: bool = True,
):
    """
    Computes representation similarity
    Example:
        >>> from torchmetrics.functional import embedding_similarity
        >>> embeddings = torch.tensor([[1., 2., 3., 4.], [1., 2., 3., 4.], [4., 5., 6., 7.]])
        >>> embedding_similarity(embeddings)
        tensor([[0.0000, 1.0000, 0.9759],
                [1.0000, 0.0000, 0.9759],
                [0.9759, 0.9759, 0.0000]])
    Args:
        batch: (batch, dim)
        similarity: 'dot' or 'cosine'
        reduction: 'none', 'sum', 'mean' (all along dim -1)
        zero_diagonal: if True, the diagonals are set to zero
    Return:
        A square matrix (batch, batch) with the similarity scores between all elements
        If sum or mean are used, then returns (b, 1) with the reduced value for each row
    """
    if similarity == "cosine":
        norm = torch.norm(batch, p=2, dim=1)
        batch = batch / norm.unsqueeze(1)

    sqr_mtx = batch.mm(batch.transpose(1, 0))

    if zero_diagonal:
        sqr_mtx = sqr_mtx.fill_diagonal_(0)

    if reduction == "mean":
        sqr_mtx = sqr_mtx.mean(dim=-1)

    if reduction == "sum":
        sqr_mtx = sqr_mtx.sum(dim=-1)

    return sqr_mtx
