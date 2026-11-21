import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pointcept/models/utils"))

from serialization.hilbert import encode as hilbert_ref
from serialization.z_order import xyz2key
from serialization.triton_encode import encode_triton


def ref(c, depth, hilbert):
    c = c.cpu().long()
    if hilbert:
        return hilbert_ref(c, num_dims=3, num_bits=depth)
    return xyz2key(c[:, 0], c[:, 1], c[:, 2], depth=depth)


@pytest.mark.parametrize("hilbert", [False, True])
def test_triton_matches_torch_all_depths(hilbert):
    for depth in range(1, 17):
        c = torch.randint(0, 1 << depth, (10_000, 3), device="cuda")
        assert torch.equal(
            encode_triton(c, depth, hilbert).cpu(), ref(c, depth, hilbert)
        ), depth


@pytest.mark.parametrize("hilbert", [False, True])
def test_out_of_range_and_negative_coords(hilbert):
    c = torch.randint(-70_000, 70_000, (10_000, 3), device="cuda")
    for depth in (4, 10, 16):
        assert torch.equal(
            encode_triton(c, depth, hilbert).cpu(), ref(c, depth, hilbert)
        ), depth


@pytest.mark.parametrize("hilbert", [False, True])
def test_int32_and_noncontiguous(hilbert):
    wide = torch.randint(0, 1 << 16, (10_000, 6), device="cuda", dtype=torch.int32)
    c = wide[:, ::2]  # strides (6, 2)
    assert torch.equal(encode_triton(c, 16, hilbert).cpu(), ref(c, 16, hilbert))
