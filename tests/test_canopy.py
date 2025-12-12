import torch
import numpy as np
import pytest
from ..torchrtm.canopy.lidf import lidf_1, lidf_2, lidf_3, lidf_4

@pytest.mark.parametrize("a, b", [(-0.3, -0.1)])
def test_lidf_1_scalar_and_batch(a, b):
    na = torch.arange(13)
    # Scalar input
    out_scalar = lidf_1(na, a=a, b=b)
    assert out_scalar.shape == (13,)
    assert torch.isclose(out_scalar.sum(), torch.tensor(1.0), atol=1e-4)

    # Batched input
    a_batch = torch.tensor([a, a]).unsqueeze(1)
    b_batch = torch.tensor([b, b]).unsqueeze(1)
    out_batch = lidf_1(na, a=a_batch, b=b_batch)
    assert out_batch.shape == (2, 13)
    assert torch.allclose(out_batch.sum(dim=1), torch.ones(2), atol=1e-4)

def test_lidf_2_scalar_and_batch():
    na = torch.arange(13)
    # Scalar ALA
    ala_scalar = torch.tensor(45.0)
    out_scalar = lidf_2(na, ala_scalar)
    assert out_scalar.shape == (13,)
    assert torch.isclose(out_scalar.sum(), torch.tensor(1.0), atol=1e-4)

    # Batched ALA
    ala_batch = torch.tensor([30.0, 60.0])
    out_batch = lidf_2(na, ala_batch)
    assert out_batch.shape == (2, 13)
    assert torch.allclose(out_batch.sum(dim=1), torch.ones(2), atol=1e-4)

def test_lidf_3_scalar_and_batch():
    na = torch.arange(13)
    # Scalar parameters
    a, b = torch.tensor(2.0), torch.tensor(5.0)
    out_scalar = lidf_3(na, a, b)
    assert out_scalar.shape == (13,)
    assert torch.isclose(out_scalar.sum(), torch.tensor(1.0), atol=1e-4)

    # Batched parameters
    a_batch = torch.tensor([2.0, 3.0])
    b_batch = torch.tensor([5.0, 2.0])
    out_batch = lidf_3(na, a_batch, b_batch)
    assert out_batch.shape == (2, 13)
    assert torch.allclose(out_batch.sum(dim=1), torch.ones(2), atol=1e-4)

def test_lidf_4_scalar_and_batch():
    na = torch.arange(13)
    # Scalar theta
    theta = torch.tensor(2.0)
    out_scalar = lidf_4(na, theta)
    assert out_scalar.shape == (13,)
    assert torch.isclose(out_scalar.sum(), torch.tensor(1.0), atol=1e-4)

    # Batched theta
    theta_batch = torch.tensor([1.0, 3.0])
    out_batch = lidf_4(na, theta_batch)
    assert out_batch.shape == (2, 13)
    assert torch.allclose(out_batch.sum(dim=1), torch.ones(2), atol=1e-4)
