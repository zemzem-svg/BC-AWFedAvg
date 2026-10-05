"""
BC-AWFedAvg Secure Aggregation
==============================

Protocol
--------
For each participating client k:

    1. Clip the local model:
           theta_k -> clip(theta_k, C)

    2. Apply client-level Gaussian DP noise:
           theta_hat_k = clip(theta_k, C) + N(0, sigma_c^2 I)

    3. Apply the final adaptive aggregation weight:
           x_k = w_k^(t) theta_hat_k

    4. Add pairwise-cancelling masks:
           theta_tilde_k =
               x_k
               + sum_{j>k} r_{k,j}
               - sum_{j<k} r_{j,k}

The coordinator only sums the masked contributions:

           X = sum_k theta_tilde_k

Because the pairwise masks cancel:

           X = sum_k w_k^(t) theta_hat_k

The coordinator can then normalize by sum_k w_k^(t).
Since the weights are normalized in BC-AWFedAvg:

           sum_k w_k^(t) = 1,

the aggregate is directly usable as the next global model.

Important
---------
This module implements the masking/cancellation layer and assumes that
pairwise secret material has already been established between clients.

For a real distributed deployment, those pairwise secrets should be obtained
through a secure key-establishment mechanism such as DH/ECDH.

For the single-machine simulation, `generate_pairwise_secrets()` can be used
to create symmetric pairwise secrets. Those secrets must be distributed to
clients privately and must NOT be given to the coordinator.

This prototype does not implement:
    - dropout recovery,
    - malicious mask-abort handling,
    - full Bonawitz server protocol,
    - distributed DH/ECDH key establishment.

It therefore validates the weighted mask-cancellation mechanism rather than
claiming to implement the complete production Bonawitz protocol.

Reference
---------
Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving
Machine Learning", CCS 2017.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections import OrderedDict
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch


# ============================================================================
# TYPES
# ============================================================================

TensorDict = OrderedDict[str, torch.Tensor]
PairwiseSecrets = Dict[int, Dict[int, bytes]]


# ============================================================================
# PAIRWISE SECRET GENERATION
# ============================================================================

def generate_pairwise_secrets(
    client_ids: Iterable[int],
    secret_size: int = 32,
) -> PairwiseSecrets:
    """
    Generate symmetric pairwise secrets for simulation.

    For every pair (i, j):

        secret[i][j] == secret[j][i]

    In a real distributed system, these values should instead be established
    privately through a DH/ECDH-style key agreement.

    IMPORTANT:
        The caller must distribute only each client's own row of the secret
        dictionary to that client. The coordinator must not receive them.

    Parameters
    ----------
    client_ids:
        Participating client identifiers.
    secret_size:
        Number of secret bytes per pair. 32 bytes = 256 bits.

    Returns
    -------
    PairwiseSecrets
        Nested dictionary:
            secrets[client_i][client_j] = shared secret bytes
    """
    ids = sorted({int(cid) for cid in client_ids})

    if len(ids) < 2:
        raise ValueError("At least two clients are required.")

    if secret_size < 16:
        raise ValueError("secret_size should be at least 16 bytes.")

    pairwise: PairwiseSecrets = {
        cid: {}
        for cid in ids
    }

    for pos_i, client_i in enumerate(ids):
        for client_j in ids[pos_i + 1:]:
            shared_secret = secrets.token_bytes(secret_size)

            pairwise[client_i][client_j] = shared_secret
            pairwise[client_j][client_i] = shared_secret

    return pairwise


# ============================================================================
# SECRET VALIDATION
# ============================================================================

def validate_pairwise_secrets(
    client_id: int,
    all_client_ids: Iterable[int],
    pairwise_secrets: Mapping[int, bytes],
) -> None:
    """
    Validate that the local client has a secret for every other participant.
    """
    cid = int(client_id)
    ids = {int(x) for x in all_client_ids}

    if cid not in ids:
        raise ValueError(
            f"client_id={cid} is not present in all_client_ids."
        )

    expected_peers = ids - {cid}
    available_peers = set(int(x) for x in pairwise_secrets.keys())

    missing = expected_peers - available_peers

    if missing:
        raise ValueError(
            f"Missing pairwise secrets for client {cid}: "
            f"{sorted(missing)}"
        )

    for peer_id in expected_peers:
        secret_value = pairwise_secrets[peer_id]

        if not isinstance(secret_value, (bytes, bytearray)):
            raise TypeError(
                f"Pairwise secret for peer {peer_id} must be bytes."
            )

        if len(secret_value) < 16:
            raise ValueError(
                f"Pairwise secret for peer {peer_id} is too short."
            )


# ============================================================================
# CRYPTOGRAPHIC MASK EXPANSION
# ============================================================================

def _mask_bytes(
    secret: bytes,
    context: bytes,
    num_bytes: int,
) -> bytes:
    """
    Expand a secret into pseudorandom bytes using HMAC-SHA256 in counter mode.

    The HMAC input includes:
        secret
        context
        counter

    The resulting byte stream is deterministic for the same secret/context,
    while remaining unpredictable to an entity that does not know the secret.
    """
    if num_bytes < 0:
        raise ValueError("num_bytes must be non-negative.")

    output = bytearray()
    counter = 0

    while len(output) < num_bytes:
        msg = (
            context
            + b"|counter="
            + counter.to_bytes(8, "big")
        )

        block = hmac.new(
            secret,
            msg,
            hashlib.sha256,
        ).digest()

        output.extend(block)
        counter += 1

    return bytes(output[:num_bytes])


def _pairwise_mask(
    round_num: int,
    client_i: int,
    client_j: int,
    tensor_name: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    shared_secret: bytes,
    mask_scale: float = 1.0,
) -> torch.Tensor:
    """
    Generate one pairwise mask for one tensor.

    The magnitude is identical for both clients in the pair, while the sign
    is opposite:

        i < j  -> +r_ij
        i > j  -> -r_ij

    Thus:

        r_ij + r_ji = 0

    IMPORTANT:
        Unlike the old implementation, the mask cannot be regenerated from
        public round/client identifiers alone. The pairwise secret is required.
    """
    if mask_scale <= 0:
        raise ValueError("mask_scale must be positive.")

    if not isinstance(shared_secret, (bytes, bytearray)):
        raise TypeError("shared_secret must be bytes.")

    numel = int(np.prod(shape, dtype=np.int64))

    context = (
        f"BC-AWFedAvg|round={int(round_num)}|"
        f"pair={min(client_i, client_j)}:{max(client_i, client_j)}|"
        f"tensor={tensor_name}|shape={tuple(shape)}|"
        f"numel={numel}"
    ).encode("utf-8")

    raw = _mask_bytes(
        secret=bytes(shared_secret),
        context=context,
        num_bytes=numel * 8,
    )

    # Interpret HMAC output as uint64 and map to [0, 1).
    uints = np.frombuffer(raw, dtype=np.uint64)

    uniform01 = (
        (uints.astype(np.float64) + 0.5)
        / 18446744073709551616.0
    )

    # Center the distribution around zero.
    values = (2.0 * uniform01 - 1.0) * float(mask_scale)

    values = values.reshape(shape)

    # Cast to exactly the tensor dtype used by both clients.
    mask = torch.from_numpy(
        np.ascontiguousarray(values)
    ).to(
        device=device,
        dtype=dtype,
    )

    # Canonical sign convention:
    # smaller client ID -> positive
    # larger client ID  -> negative
    if client_i > client_j:
        mask = -mask

    return mask


# ============================================================================
# MODEL CLIPPING
# ============================================================================

def clip_model(
    params: Mapping[str, torch.Tensor],
    clip_norm: float,
) -> TensorDict:
    """
    Clip the complete model to a global L2 norm.

    The same clipping factor is applied to all tensors.

        theta_hat = clip(theta, C)
    """
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive.")

    squared_norm = torch.zeros(
        (),
        dtype=torch.float64,
        device=next(iter(params.values())).device,
    )

    for tensor in params.values():
        x = tensor.detach().float()
        squared_norm += torch.sum(x.double() * x.double())

    total_norm = torch.sqrt(squared_norm)

    if total_norm <= clip_norm:
        scale = 1.0
    else:
        scale = float(clip_norm / (total_norm.item() + 1e-12))

    clipped = OrderedDict()

    for name, tensor in params.items():
        clipped[name] = (
            tensor.detach()
            .float()
            .mul(scale)
        )

    return clipped


# ============================================================================
# CLIENT-LEVEL DIFFERENTIAL PRIVACY
# ============================================================================

def add_gaussian_noise(
    params: Mapping[str, torch.Tensor],
    noise_std: float,
) -> TensorDict:
    """
    Add Gaussian noise independently to each model parameter tensor.

        theta_hat = theta_clipped + N(0, sigma_c^2 I)
    """
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative.")

    protected = OrderedDict()

    for name, tensor in params.items():
        if noise_std == 0:
            noise = torch.zeros_like(tensor)
        else:
            noise = torch.randn_like(tensor) * float(noise_std)

        protected[name] = tensor + noise

    return protected


# ============================================================================
# APPLY ADAPTIVE WEIGHT
# ============================================================================

def apply_aggregation_weight(
    params: Mapping[str, torch.Tensor],
    weight: float,
) -> TensorDict:
    """
    Multiply the protected client model by its final adaptive aggregation
    weight before secure masking.

        x_k = w_k^(t) theta_hat_k
    """
    if weight < 0:
        raise ValueError("Aggregation weight must be non-negative.")

    weighted = OrderedDict()

    for name, tensor in params.items():
        weighted[name] = tensor * float(weight)

    return weighted


# ============================================================================
# ADD PAIRWISE MASKS
# ============================================================================

def add_pairwise_masks(
    params: Mapping[str, torch.Tensor],
    client_id: int,
    all_client_ids: List[int],
    round_num: int,
    pairwise_secrets: Mapping[int, bytes],
    mask_scale: float = 1.0,
) -> TensorDict:
    """
    Add pairwise-cancelling masks to an already weighted protected update.

        theta_tilde_k =
            w_k theta_hat_k
            + sum_{j>k} r_kj
            - sum_{j<k} r_jk
    """
    validate_pairwise_secrets(
        client_id=client_id,
        all_client_ids=all_client_ids,
        pairwise_secrets=pairwise_secrets,
    )

    masked = OrderedDict()

    client_id = int(client_id)
    ids = sorted(int(x) for x in all_client_ids)

    for name, tensor in params.items():

        accumulator = tensor.detach().clone()

        for peer_id in ids:
            if peer_id == client_id:
                continue

            shared_secret = pairwise_secrets[peer_id]

            mask = _pairwise_mask(
                round_num=round_num,
                client_i=client_id,
                client_j=peer_id,
                tensor_name=name,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                device=tensor.device,
                shared_secret=shared_secret,
                mask_scale=mask_scale,
            )

            accumulator = accumulator + mask

        masked[name] = accumulator

    return masked


# ============================================================================
# COMPLETE CLIENT-SIDE PROTECTION
# ============================================================================

def protect_and_mask(
    params: Mapping[str, torch.Tensor],
    client_id: int,
    all_client_ids: List[int],
    round_num: int,
    weight: float,
    pairwise_secrets: Mapping[int, bytes],
    clip_norm: float = 1.0,
    noise_std: float = 0.0,
    mask_scale: float = 1.0,
) -> TensorDict:
    """
    Complete BC-AWFedAvg client-side protection pipeline.

        theta_k
            ↓
        clip(theta_k, C)
            ↓
        + Gaussian DP noise
            ↓
        w_k^(t) *
            ↓
        pairwise masks
            ↓
        theta_tilde_k

    This is the function that should normally be called by the client.
    """
    clipped = clip_model(
        params=params,
        clip_norm=clip_norm,
    )

    protected = add_gaussian_noise(
        params=clipped,
        noise_std=noise_std,
    )

    weighted = apply_aggregation_weight(
        params=protected,
        weight=weight,
    )

    masked = add_pairwise_masks(
        params=weighted,
        client_id=client_id,
        all_client_ids=all_client_ids,
        round_num=round_num,
        pairwise_secrets=pairwise_secrets,
        mask_scale=mask_scale,
    )

    return masked


# ============================================================================
# MASKED AGGREGATION AT COORDINATOR
# ============================================================================

def aggregate_masked_parameters(
    masked_updates: List[Mapping[str, torch.Tensor]],
    output_dtype: Optional[torch.dtype] = None,
) -> TensorDict:
    """
    Sum masked weighted client contributions.

        sum_k theta_tilde_k
          = sum_k w_k theta_hat_k

    No individual unmasked contribution is reconstructed here.

    Parameters
    ----------
    masked_updates:
        One masked OrderedDict per participating client.

    output_dtype:
        Optional output dtype. If None, uses dtype of the first tensor.

    Returns
    -------
    TensorDict
        Aggregated weighted protected model.
    """
    if not masked_updates:
        raise ValueError("masked_updates must not be empty.")

    reference_names = list(masked_updates[0].keys())

    for update in masked_updates:
        if list(update.keys()) != reference_names:
            raise ValueError("All clients must use identical parameter keys.")

    aggregated = OrderedDict()

    for name in reference_names:

        # Accumulate in float64 to reduce residual numerical error from
        # pairwise cancellation.
        first = masked_updates[0][name]

        accumulator = first.detach().double().clone()

        for update in masked_updates[1:]:
            accumulator += update[name].detach().double()

        if output_dtype is None:
            aggregated[name] = accumulator.to(first.dtype)
        else:
            aggregated[name] = accumulator.to(output_dtype)

    return aggregated


# ============================================================================
# MASK-CANCELLATION VERIFICATION
# ============================================================================

def verify_mask_cancellation(
    client_ids: List[int],
    round_num: int,
    tensor_shapes: Mapping[str, Tuple[int, ...]],
    pairwise_secrets: PairwiseSecrets,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    mask_scale: float = 1.0,
    atol: float = 1e-5,
) -> bool:
    """
    Functional unit test for pairwise mask cancellation.

    This verifies:

        sum_k M_k = 0

    It does NOT constitute a cryptographic security proof.
    """
    ids = sorted(int(x) for x in client_ids)

    if device is None:
        device = torch.device("cpu")

    for tensor_name, shape in tensor_shapes.items():

        total_mask = torch.zeros(
            shape,
            dtype=dtype,
            device=device,
        )

        for client_id in ids:
            validate_pairwise_secrets(
                client_id=client_id,
                all_client_ids=ids,
                pairwise_secrets=pairwise_secrets[client_id],
            )

            for peer_id in ids:
                if peer_id == client_id:
                    continue

                mask = _pairwise_mask(
                    round_num=round_num,
                    client_i=client_id,
                    client_j=peer_id,
                    tensor_name=tensor_name,
                    shape=shape,
                    dtype=dtype,
                    device=device,
                    shared_secret=pairwise_secrets[client_id][peer_id],
                    mask_scale=mask_scale,
                )

                total_mask += mask

        if not torch.allclose(
            total_mask,
            torch.zeros_like(total_mask),
            atol=atol,
            rtol=0.0,
        ):
            return False

    return True


# ============================================================================
# WEIGHTED AGGREGATION VERIFICATION
# ============================================================================

def verify_weighted_secure_aggregation(
    params_by_client: Mapping[int, TensorDict],
    weights: Mapping[int, float],
    round_num: int = 1,
    clip_norm: float = 1.0,
    noise_std: float = 0.0,
    mask_scale: float = 1.0,
) -> bool:
    """
    Functional test showing that secure masking preserves the weighted
    aggregate.

    The test compares:

        sum_k masked_k

    against:

        sum_k w_k * protected_k

    It is intended for development/unit testing only.
    """
    client_ids = sorted(int(x) for x in params_by_client.keys())

    if set(client_ids) != set(int(x) for x in weights.keys()):
        raise ValueError(
            "params_by_client and weights must contain the same clients."
        )

    pairwise_secrets = generate_pairwise_secrets(client_ids)

    masked_updates: List[TensorDict] = []
    direct_updates: List[TensorDict] = []

    for client_id in client_ids:

        clipped = clip_model(
            params_by_client[client_id],
            clip_norm=clip_norm,
        )

        protected = add_gaussian_noise(
            clipped,
            noise_std=noise_std,
        )

        weighted = apply_aggregation_weight(
            protected,
            weight=float(weights[client_id]),
        )

        direct_updates.append(weighted)

        masked = add_pairwise_masks(
            weighted,
            client_id=client_id,
            all_client_ids=client_ids,
            round_num=round_num,
            pairwise_secrets=pairwise_secrets[client_id],
            mask_scale=mask_scale,
        )

        masked_updates.append(masked)

    secure_sum = aggregate_masked_parameters(masked_updates)

    direct_sum = OrderedDict()

    for name in direct_updates[0].keys():
        value = direct_updates[0][name].detach().double().clone()

        for update in direct_updates[1:]:
            value += update[name].detach().double()

        direct_sum[name] = value.to(secure_sum[name].dtype)

    for name in direct_sum.keys():
        if not torch.allclose(
            secure_sum[name],
            direct_sum[name],
            atol=1e-5,
            rtol=1e-5,
        ):
            return False

    return True


# ============================================================================
# SIMPLE CONVERSION HELPERS
# ============================================================================

def ordered_dict_to_numpy(
    params: Mapping[str, torch.Tensor],
) -> List[np.ndarray]:
    """
    Convert an OrderedDict of tensors to NumPy arrays.
    """
    return [
        tensor.detach().cpu().numpy()
        for tensor in params.values()
    ]


def numpy_to_ordered_dict(
    arrays: List[np.ndarray],
    keys: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> TensorDict:
    """
    Convert NumPy arrays to an OrderedDict of float32 tensors.
    """
    if device is None:
        device = torch.device("cpu")

    if keys is None:
        keys = [
            f"param_{i}"
            for i in range(len(arrays))
        ]

    if len(keys) != len(arrays):
        raise ValueError(
            "Number of keys must match number of arrays."
        )

    return OrderedDict(
        (
            key,
            torch.as_tensor(
                array,
                dtype=torch.float32,
                device=device,
            ),
        )
        for key, array in zip(keys, arrays)
    )
    # """
# Secure Aggregation — Mask-Based Pairwise Cancellation
# ======================================================

# Principle
# ---------
# Each client k adds a mask derived from pairwise shared seeds:

#     w̃_k = w_k + Σ_{j>k} r_{kj} - Σ_{j<k} r_{jk}

# where r_{kj} is a pseudo-random mask shared between client k and j.

# When the coordinator sums all masked updates:

#     Σ_k w̃_k = Σ_k w_k      (masks cancel pairwise)

# The coordinator learns only the aggregate — never individual updates.

# Implementation
# --------------
# For simplicity (single-machine simulation) the masks are generated
# deterministically from a round seed and client-pair indices, which is
# equivalent to the pairwise DH-exchange model but without network overhead.

# Reference: Bonawitz et al., "Practical Secure Aggregation for
# Privacy-Preserving Machine Learning", CCS 2017.
# """

# from __future__ import annotations

# import hashlib
# from typing import List, Dict
# from collections import OrderedDict

# import numpy as np
# import torch


# def _pairwise_mask(
#     round_num: int,
#     client_i: int,
#     client_j: int,
#     shape: tuple,
#     dtype: torch.dtype,
# ) -> torch.Tensor:
#     """
#     Generate a deterministic pseudo-random mask for the (i, j) pair.
#     r_{ij} = -r_{ji} by construction (sign flips).
#     """
#     # Canonical ordering: always hash (min, max) so r_{ij} == r_{ji} in magnitude
#     lo, hi = min(client_i, client_j), max(client_i, client_j)
#     seed_str = f"secagg|round={round_num}|pair=({lo},{hi})"
#     seed_int  = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % (2**31)
#     rng = torch.Generator()
#     rng.manual_seed(seed_int)
#     mask = torch.randn(shape, generator=rng, dtype=dtype)
#     # Sign: client_i adds +mask if i < j, else subtracts
#     return mask if client_i < client_j else -mask


# def add_secure_mask(
#     params: OrderedDict,
#     client_id: int,
#     all_client_ids: List[int],
#     round_num: int,
# ) -> OrderedDict:
#     """
#     Add pairwise-cancelling masks to client parameters before sending to server.

#     Parameters
#     ----------
#     params          : local model parameters (OrderedDict of tensors)
#     client_id       : this client's integer id
#     all_client_ids  : list of all participating client ids this round
#     round_num       : current FL round number (keeps masks round-specific)

#     Returns
#     -------
#     masked_params   : params + Σ masks  (coordinator cannot invert without all masks)
#     """
#     masked = OrderedDict()
#     for name, tensor in params.items():
#         acc = tensor.clone().float()
#         for j in all_client_ids:
#             if j == client_id:
#                 continue
#             mask = _pairwise_mask(round_num, client_id, j, tensor.shape, torch.float32)
#             acc = acc + mask
#         masked[name] = acc.to(tensor.dtype)
#     return masked


# def verify_mask_cancellation(
#     num_clients: int,
#     round_num: int,
#     shape: tuple = (4,),
# ) -> bool:
#     """
#     Sanity-check: sum of all masks for a given parameter shape is ≈ 0.
#     Useful in unit tests.
#     """
#     ids = list(range(num_clients))
#     total = torch.zeros(shape)
#     for cid in ids:
#         for j in ids:
#             if j == cid:
#                 continue
#             total += _pairwise_mask(round_num, cid, j, shape, torch.float32)
#     return bool(torch.allclose(total, torch.zeros(shape), atol=1e-5))
