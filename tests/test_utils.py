import torch
import numpy as np
from ..torchrtm.utils import (
    exp1,
    betainc
)


def test_exp1_accuracy():
    x = torch.tensor([0.1, 1.0, 5.0])
    out = exp1(x)
    assert out.shape == x.shape
    assert torch.all(out > 0)


