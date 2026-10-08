"""
BC-AWFedAvg Secure Aggregation
==============================

Weighted pairwise-mask secure aggregation used by the BC-AWFedAvg prototype.

Protocol
--------
The client computes, in this order:

    theta_hat_k = Clip(theta_k, C) + N(0, sigma_c^2 I)
    x_k         = w_k^(t) * theta_hat_k
    theta_tilde = x_k + M_k

where

    M_k = sum_{j>k} r_{k,j} - sum_{j<k} r_{j,k}

and, for every participating pair,

    r_{k,j} = -r_{j,k}.

Therefore, when the coordinator sums the masked contributions,

    sum_k theta_tilde_k = sum_k w_k^(t) theta_hat_k.

Important security note
-----------------------
For the single-machine research prototype, pairwise secrets are generated once
and stored only by the simulated clients. This reproduces the pairwise-secret
masking property but is NOT a complete Bonawitz et al. deployment: there is no
networked DH/ECDH key-establishment protocol, dropout recovery, or malicious
mask-abort recovery.

The coordinator must never receive the pairwise secret dictionary in a real
federated deployment.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets as _secrets
from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch

TensorDict = OrderedDict[str, torch.Tensor]
PairwiseSecretStore = Dict[int, Dict[int, bytes]]


def generate_pairwise_secrets(
    client_ids: Iterable[int],
    secret_size: int = 32,
) -> PairwiseSecretStore:
    """Create one independent 256-bit secret for every client pair.

    In production these pairwise secrets should be established by a proper
    authenticated DH/ECDH protocol. Here they are generated once for the local
    simulation.
    """
    ids = sorted({int(cid) for cid in client_ids})
    if len(ids) < 2:
        raise ValueError("At least two clients are required.")
    if secret_size < 16:
        raise ValueError("Pairwise secret must contain at least 16 bytes.")

    store: PairwiseSecretStore = {cid: {} for cid in ids}
    for i, cid_i in enumerate(ids):
        for cid_j in ids[i + 1 :]:
            shared = _secrets.token_bytes(secret_size)
            store[cid_i][cid_j] = shared
            store[cid_j][cid_i] = shared
    return store


def _expand_secret(secret: bytes, context: bytes, nbytes: int) -> bytes:
    """Expand a pairwise secret into pseudorandom bytes with HMAC-SHA256."""
    output = bytearray()
    counter = 0
    while len(output) < nbytes:
        msg = context + counter.to_bytes(8, "big")
        output.extend(hmac.new(secret, msg, hashlib.sha256).digest())
        counter += 1
    return bytes(output[:nbytes])


def _pairwise_mask(
    round_num: int,
    client_i: int,
    client_j: int,
    tensor_name: str,
    shape: Tuple[int, ...],
    device: torch.device,
    shared_secret: bytes,
    mask_scale: float = 1.0,
) -> torch.Tensor:
    """Generate a pairwise mask using secret shared randomness.

    The same magnitude is produced at both endpoints; the sign is opposite.
    The mask values are materialized as float32 so the two endpoints use the
    same numerical values and pairwise cancellation is exact up to the final
    floating-point summation.
    """
    if not isinstance(shared_secret, (bytes, bytearray)):
        raise TypeError("shared_secret must be bytes.")
    if mask_scale <= 0:
        raise ValueError("mask_scale must be positive.")

    numel = int(np.prod(shape, dtype=np.int64))
    lo, hi = sorted((int(client_i), int(client_j)))
    context = (
        f"BC-AWFedAvg|r={int(round_num)}|pair={lo}:{hi}|"
        f"tensor={tensor_name}|shape={shape}|n={numel}"
    ).encode("utf-8")

    raw = _expand_secret(shared_secret, context, numel * 8)
    values64 = np.frombuffer(raw, dtype=np.uint64).astype(np.float64)
    values64 = (values64 + 0.5) / 18446744073709551616.0
    values32 = ((2.0 * values64 - 1.0) * float(mask_scale)).astype(np.float32)
    values32 = values32.reshape(shape)

    mask = torch.from_numpy(np.ascontiguousarray(values32)).to(
        device=device,
        dtype=torch.float32,
    )
    if client_i > client_j:
        mask = -mask
    return mask


def validate_pairwise_secrets(
    client_id: int,
    all_client_ids: Iterable[int],
    pairwise_secrets: Mapping[int, bytes],
) -> None:
    """Ensure that one client has a secret for every other participant."""
    cid = int(client_id)
    ids = {int(x) for x in all_client_ids}
    if cid not in ids:
        raise ValueError(f"Client {cid} is not in the participant set.")

    missing = (ids - {cid}) - {int(x) for x in pairwise_secrets.keys()}
    if missing:
        raise ValueError(f"Client {cid} is missing secrets for {sorted(missing)}")

    for peer, secret in pairwise_secrets.items():
        if not isinstance(secret, (bytes, bytearray)) or len(secret) < 16:
            raise ValueError(f"Invalid pairwise secret for peer {peer}.")


def add_pairwise_masks(
    params: Mapping[str, torch.Tensor],
    client_id: int,
    all_client_ids: List[int],
    round_num: int,
    pairwise_secrets: Mapping[int, bytes],
    mask_scale: float = 1.0,
) -> TensorDict:
    """Mask an already protected and weighted client contribution."""
    validate_pairwise_secrets(client_id, all_client_ids, pairwise_secrets)

    ids = sorted(int(x) for x in all_client_ids)
    masked = OrderedDict()

    for name, tensor in params.items():
        base = tensor.detach().float().clone()
        acc = base.clone()

        for peer_id in ids:
            if peer_id == int(client_id):
                continue
            mask = _pairwise_mask(
                round_num=round_num,
                client_i=int(client_id),
                client_j=peer_id,
                tensor_name=name,
                shape=tuple(base.shape),
                device=base.device,
                shared_secret=pairwise_secrets[peer_id],
                mask_scale=mask_scale,
            )
            acc.add_(mask)

        masked[name] = acc.to(dtype=tensor.dtype)

    return masked


def aggregate_masked_parameters(
    masked_updates: List[Mapping[str, torch.Tensor]],
    output_dtype: Optional[torch.dtype] = None,
) -> TensorDict:
    """Sum masked weighted contributions without reconstructing individuals."""
    if not masked_updates:
        raise ValueError("masked_updates must not be empty.")

    names = list(masked_updates[0].keys())
    for update in masked_updates:
        if list(update.keys()) != names:
            raise ValueError("All masked updates must have identical parameter keys.")

    aggregated = OrderedDict()
    for name in names:
        first = masked_updates[0][name]
        acc = first.detach().double().clone()
        for update in masked_updates[1:]:
            acc.add_(update[name].detach().double())
        aggregated[name] = acc.to(output_dtype or first.dtype)

    return aggregated


def verify_mask_cancellation(
    client_ids: List[int],
    round_num: int,
    tensor_shapes: Mapping[str, Tuple[int, ...]],
    pairwise_secrets: PairwiseSecretStore,
    mask_scale: float = 1.0,
    atol: float = 1e-5,
) -> bool:
    """Unit test for the algebraic property sum_k M_k = 0."""
    ids = sorted(int(x) for x in client_ids)

    for name, shape in tensor_shapes.items():
        total = torch.zeros(shape, dtype=torch.float32)
        for cid in ids:
            row = pairwise_secrets[cid]
            for peer in ids:
                if peer == cid:
                    continue
                total += _pairwise_mask(
                    round_num=round_num,
                    client_i=cid,
                    client_j=peer,
                    tensor_name=name,
                    shape=shape,
                    device=torch.device("cpu"),
                    shared_secret=row[peer],
                    mask_scale=mask_scale,
                )
        if not torch.allclose(total, torch.zeros_like(total), atol=atol, rtol=0.0):
            return False

    return True


def verify_weighted_secure_aggregation(
    weighted_updates: Mapping[int, TensorDict],
    round_num: int = 1,
    mask_scale: float = 1.0,
) -> bool:
    """Verify that masking preserves an already weighted sum.

    This is a correctness test, not a cryptographic security proof.
    """
    ids = sorted(int(x) for x in weighted_updates.keys())
    secrets = generate_pairwise_secrets(ids)
    masked = []

    for cid in ids:
        masked.append(
            add_pairwise_masks(
                params=weighted_updates[cid],
                client_id=cid,
                all_client_ids=ids,
                round_num=round_num,
                pairwise_secrets=secrets[cid],
                mask_scale=mask_scale,
            )
        )

    secured = aggregate_masked_parameters(masked, output_dtype=torch.float32)

    direct = OrderedDict()
    for name in weighted_updates[ids[0]].keys():
        value = weighted_updates[ids[0]][name].double().clone()
        for cid in ids[1:]:
            value += weighted_updates[cid][name].double()
        direct[name] = value.float()

    return all(
        torch.allclose(secured[name], direct[name], atol=1e-5, rtol=1e-5)
        for name in direct
    )
