"""
Secure Aggregation — Mask-Based Pairwise Cancellation
======================================================

Principle
---------
Each client k adds a mask derived from pairwise shared seeds:

    w̃_k = w_k + Σ_{j>k} r_{kj} - Σ_{j<k} r_{jk}

where r_{kj} is a pseudo-random mask shared between client k and j.

When the coordinator sums all masked updates:

    Σ_k w̃_k = Σ_k w_k      (masks cancel pairwise)

The coordinator learns only the aggregate — never individual updates.

Implementation
--------------
For simplicity (single-machine simulation) the masks are generated
deterministically from a round seed and client-pair indices, which is
equivalent to the pairwise DH-exchange model but without network overhead.

Reference: Bonawitz et al., "Practical Secure Aggregation for
Privacy-Preserving Machine Learning", CCS 2017.
"""

from __future__ import annotations

import hashlib
from typing import List, Dict
from collections import OrderedDict

import numpy as np
import torch


def _pairwise_mask(
    round_num: int,
    client_i: int,
    client_j: int,
    shape: tuple,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Generate a deterministic pseudo-random mask for the (i, j) pair.
    r_{ij} = -r_{ji} by construction (sign flips).
    """
    # Canonical ordering: always hash (min, max) so r_{ij} == r_{ji} in magnitude
    lo, hi = min(client_i, client_j), max(client_i, client_j)
    seed_str = f"secagg|round={round_num}|pair=({lo},{hi})"
    seed_int  = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**31)
    rng = torch.Generator()
    rng.manual_seed(seed_int)
    mask = torch.randn(shape, generator=rng, dtype=dtype)
    # Sign: client_i adds +mask if i < j, else subtracts
    return mask if client_i < client_j else -mask


def add_secure_mask(
    params: OrderedDict,
    client_id: int,
    all_client_ids: List[int],
    round_num: int,
) -> OrderedDict:
    """
    Add pairwise-cancelling masks to client parameters before sending to server.

    Parameters
    ----------
    params          : local model parameters (OrderedDict of tensors)
    client_id       : this client's integer id
    all_client_ids  : list of all participating client ids this round
    round_num       : current FL round number (keeps masks round-specific)

    Returns
    -------
    masked_params   : params + Σ masks  (coordinator cannot invert without all masks)
    """
    masked = OrderedDict()
    for name, tensor in params.items():
        acc = tensor.clone().float()
        for j in all_client_ids:
            if j == client_id:
                continue
            mask = _pairwise_mask(round_num, client_id, j, tensor.shape, torch.float32)
            acc = acc + mask
        masked[name] = acc.to(tensor.dtype)
    return masked


def verify_mask_cancellation(
    num_clients: int,
    round_num: int,
    shape: tuple = (4,),
) -> bool:
    """
    Sanity-check: sum of all masks for a given parameter shape is ≈ 0.
    Useful in unit tests.
    """
    ids = list(range(num_clients))
    total = torch.zeros(shape)
    for cid in ids:
        for j in ids:
            if j == cid:
                continue
            total += _pairwise_mask(round_num, cid, j, shape, torch.float32)
    return bool(torch.allclose(total, torch.zeros(shape), atol=1e-5))
