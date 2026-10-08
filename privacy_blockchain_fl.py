"""
Blockchain/IPFS governance and encrypted publication utilities for BC-AWFedAvg.

This module deliberately does NOT implement client-update differential privacy.
Client-update DP belongs to ``efficient_dp.py`` and is applied before adaptive
weighting and pairwise Secure Aggregation.

Responsibilities of this module:
    - Ethereum/Ganache smart-contract interaction
    - client registration and contribution metadata
    - reputation validation/update calls
    - encrypted off-chain model publication through IPFS
    - publication-level server DP for the public/audit copy

Canonical BC-AWFedAvg learning path:
    local model -> client update -> client DP -> adaptive weight
    -> pairwise masks -> secure sum -> live global model
    -> optional publication DP -> encrypt -> IPFS -> blockchain metadata
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
import warnings
from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

try:
    import ipfshttpclient
except ImportError:  # pragma: no cover - only needed when IPFS is used
    ipfshttpclient = None

from web3 import Web3

try:
    from web3.middleware import ExtraDataToPOAMiddleware
except ImportError:  # web3.py < 7
    ExtraDataToPOAMiddleware = None

try:
    from web3.middleware import geth_poa_middleware
except ImportError:  # pragma: no cover - web3.py 7+
    geth_poa_middleware = None

from efficient_dp import EfficientDPManager

logger = logging.getLogger(__name__)

AES_VERSION = b"BC-AWFEDAVG-AESGCM1"
AES_NONCE_SIZE = 12
RSA_KEY_SIZE = 4096


class _UnlockedAccount:
    """Small account-like object for an unlocked local Ganache account."""

    def __init__(self, address: str):
        self.address = address


class PrivacyPreservingFederatedLearning:
    """Governance, encryption, IPFS, and publication-DP backend for BC-AWFedAvg."""

    def __init__(
        self,
        blockchain_provider: str = "http://127.0.0.1:8545",
        contract_address: Optional[str] = None,
        contract_abi_path: Optional[str] = None,
        ipfs_addr: str = "/ip4/127.0.0.1/tcp/5001",
        epsilon: float = 1.0,
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        coordinator_private_key: Optional[str] = None,
        require_connection: bool = True,
    ) -> None:
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if not (0.0 < delta < 1.0):
            raise ValueError("delta must satisfy 0 < delta < 1.")
        if clip_norm <= 0:
            raise ValueError("clip_norm must be positive.")

        self.blockchain_provider = blockchain_provider
        self.contract_address = contract_address
        self._contract_abi_path = contract_abi_path
        self._ipfs_addr = ipfs_addr
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.clip_norm = float(clip_norm)

        self.w3 = Web3(Web3.HTTPProvider(blockchain_provider))
        self._inject_poa_middleware()

        connected = self.w3.is_connected()
        if require_connection and not connected:
            raise ConnectionError(
                f"Failed to connect to blockchain at {blockchain_provider}. "
                "Start Ganache or disable blockchain mode."
            )
        if not connected:
            warnings.warn(
                f"[PPFL] Blockchain node not reachable at {blockchain_provider}; offline mode.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.contract = None
        if connected and contract_address and contract_abi_path:
            self.contract = self._load_contract(contract_address, contract_abi_path)

        self.ipfs = None
        if ipfshttpclient is not None:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=DeprecationWarning)
                    try:
                        self.ipfs = ipfshttpclient.connect(ipfs_addr, check_version=False)
                    except TypeError:
                        self.ipfs = ipfshttpclient.connect(ipfs_addr)
                logger.info("Connected to IPFS at %s", ipfs_addr)
            except Exception as exc:
                warnings.warn(f"[PPFL] IPFS connection failed: {exc}", RuntimeWarning)
        elif require_connection:
            warnings.warn(
                "[PPFL] ipfshttpclient is not installed; IPFS functions are unavailable.",
                RuntimeWarning,
            )

        self.coordinator_account = self._load_coordinator_account(coordinator_private_key)
        self.client_keys: Dict[str, Dict[str, bytes]] = {}

        # Publication-level DP is separate from the client-update DP accountant.
        self.publication_dp = EfficientDPManager(
            epsilon=self.epsilon,
            delta=self.delta,
            initial_clip_norm=self.clip_norm,
            adaptive_clip=False,
        )

        self.performance_metrics: Dict[str, list] = {
            "upload_times": [],
            "download_times": [],
            "encryption_times": [],
            "decryption_times": [],
            "model_sizes": [],
            "serialized_sizes": [],
            "privacy_overhead": [],
            "blockchain_times": [],
        }

    # ------------------------------------------------------------------
    # Connection / ABI helpers
    # ------------------------------------------------------------------

    def _inject_poa_middleware(self) -> None:
        try:
            if ExtraDataToPOAMiddleware is not None:
                self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
            elif geth_poa_middleware is not None:
                self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        except Exception as exc:
            logger.debug("PoA middleware injection skipped: %s", exc)

    def _load_contract(self, address: str, abi_path: str):
        with open(abi_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict):
            abi = raw.get("abi") or raw.get("contract_abi")
        else:
            abi = raw
        if not isinstance(abi, list):
            raise ValueError("Contract ABI JSON must contain 'abi' or 'contract_abi'.")

        contract = self.w3.eth.contract(
            address=Web3.to_checksum_address(address),
            abi=abi,
        )
        required = {
            "startRound",
            "submitAggregatedModel",
            "getClientInfo",
            "recordContributionsBatch",
            "updateReputationsBatch",
        }
        available = {
            item.get("name")
            for item in abi
            if item.get("type") == "function"
        }
        missing = required - available
        if missing:
            raise ValueError(f"Contract ABI is missing required functions: {sorted(missing)}")
        return contract

    def _load_coordinator_account(self, private_key: Optional[str]):
        if not self.w3.is_connected():
            return None
        if private_key:
            account = self.w3.eth.account.from_key(private_key)
            return account
        accounts = self.w3.eth.accounts
        return _UnlockedAccount(accounts[0]) if accounts else None

    # ------------------------------------------------------------------
    # Transaction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _raw_signed_transaction(signed_tx: Any) -> bytes:
        return getattr(signed_tx, "raw_transaction", None) or getattr(
            signed_tx, "rawTransaction", None
        )

    def _send_function(
        self,
        fn: Any,
        sender: str,
        private_key: Optional[str] = None,
        value: int = 0,
        gas: int = 300_000,
    ) -> str:
        """Send a contract function using a key or an explicitly unlocked account."""
        sender = Web3.to_checksum_address(sender)
        if private_key:
            account = self.w3.eth.account.from_key(private_key)
            if account.address.lower() != sender.lower():
                raise ValueError("Private key does not correspond to sender address.")
            tx = fn.build_transaction(
                {
                    "from": sender,
                    "nonce": self.w3.eth.get_transaction_count(sender),
                    "gas": gas,
                    "gasPrice": self.w3.eth.gas_price,
                    "value": int(value),
                    "chainId": self.w3.eth.chain_id,
                }
            )
            signed = account.sign_transaction(tx)
            raw = self._raw_signed_transaction(signed)
            tx_hash = self.w3.eth.send_raw_transaction(raw)
        else:
            available = {a.lower() for a in self.w3.eth.accounts}
            if sender.lower() not in available:
                raise RuntimeError(
                    f"Sender {sender} is not an unlocked local account and no private key was supplied."
                )
            tx_hash = fn.transact(
                {
                    "from": sender,
                    "gas": gas,
                    "value": int(value),
                }
            )

        self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return tx_hash.hex()

    # ------------------------------------------------------------------
    # Key management
    # ------------------------------------------------------------------

    def generate_client_keypair(self, client_id: str) -> Tuple[bytes, bytes]:
        """Create an RSA-4096 key pair once per client identity and persist in memory."""
        client_id = str(client_id)
        if client_id in self.client_keys:
            item = self.client_keys[client_id]
            return item["private"], item["public"]

        private_key = rsa.generate_private_key(public_exponent=65537, key_size=RSA_KEY_SIZE)
        public_key = private_key.public_key()
        private_pem = private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        public_pem = public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        self.client_keys[client_id] = {"private": private_pem, "public": public_pem}
        return private_pem, public_pem

    @staticmethod
    def get_public_key_hash(public_key_pem: bytes) -> bytes:
        return hashlib.sha256(public_key_pem).digest()

    # ------------------------------------------------------------------
    # Encrypted model publication
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_model(model_params: Mapping[str, torch.Tensor]) -> bytes:
        arrays = {
            str(name): value.detach().cpu().to(torch.float16).numpy()
            for name, value in model_params.items()
            if torch.is_tensor(value)
        }
        if not arrays:
            raise ValueError("No tensor model parameters were provided.")
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return buf.getvalue()

    def encrypt_model(
        self,
        model_params: Mapping[str, torch.Tensor],
        compression: bool = True,
    ) -> Tuple[bytes, bytes]:
        """Compress then encrypt model parameters with AES-256-GCM.

        The returned symmetric key is 32 bytes and can be wrapped for a
        recipient with ``encrypt_symmetric_key``.
        """
        start = time.time()
        model_bytes = self._serialize_model(model_params) if compression else self._serialize_model(model_params)
        self.performance_metrics["serialized_sizes"].append(len(model_bytes))

        key = AESGCM.generate_key(bit_length=256)
        nonce = os.urandom(AES_NONCE_SIZE)
        ciphertext = AESGCM(key).encrypt(nonce, model_bytes, AES_VERSION)
        payload = AES_VERSION + nonce + ciphertext

        elapsed = time.time() - start
        self.performance_metrics["encryption_times"].append(elapsed)
        self.performance_metrics["model_sizes"].append(len(payload))
        return payload, key

    def decrypt_model(
        self,
        encrypted_data: bytes,
        symmetric_key: bytes,
        compression: bool = True,
    ) -> OrderedDict:
        del compression
        start = time.time()
        prefix_len = len(AES_VERSION)
        if not encrypted_data.startswith(AES_VERSION):
            raise ValueError("Unsupported or malformed encrypted BC-AWFedAvg payload.")
        nonce_start = prefix_len
        nonce_end = nonce_start + AES_NONCE_SIZE
        nonce = encrypted_data[nonce_start:nonce_end]
        ciphertext = encrypted_data[nonce_end:]
        if len(symmetric_key) != 32:
            raise ValueError("AES-256-GCM key must contain 32 bytes.")

        model_bytes = AESGCM(symmetric_key).decrypt(nonce, ciphertext, AES_VERSION)
        npz = np.load(io.BytesIO(model_bytes), allow_pickle=False)
        model = OrderedDict(
            (name, torch.tensor(npz[name].astype(np.float32)))
            for name in npz.files
        )
        self.performance_metrics["decryption_times"].append(time.time() - start)
        return model

    def encrypt_symmetric_key(self, symmetric_key: bytes, public_key_pem: bytes) -> bytes:
        public_key = serialization.load_pem_public_key(public_key_pem)
        return public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    def decrypt_symmetric_key(self, encrypted_key: bytes, private_key_pem: bytes) -> bytes:
        private_key = serialization.load_pem_private_key(private_key_pem, password=None)
        return private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

    # ------------------------------------------------------------------
    # Publication-level server DP
    # ------------------------------------------------------------------

    def add_publication_dp_noise(
        self,
        global_model: Mapping[str, torch.Tensor],
        sensitivity: float = 1.0,
    ) -> Tuple[OrderedDict, float]:
        """Add server-side Gaussian noise to the public/audit copy.

        This method does not clip the global model and does not alter the live
        federated model. Its privacy accounting is kept separate from
        client-update DP.
        """
        start = time.time()
        C = float(sensitivity)
        if C <= 0:
            raise ValueError("sensitivity must be positive.")
        sigma = self.publication_dp.noise_sigma(sensitivity=C)
        protected = OrderedDict()
        for name, value in global_model.items():
            if torch.is_tensor(value):
                protected[name] = value.detach().float() + torch.randn_like(value.float()) * sigma
            else:
                protected[name] = value
        self.publication_dp.rdp.step(noise_sigma=sigma, sensitivity=C)
        self.performance_metrics["privacy_overhead"].append(time.time() - start)
        return protected, sigma

    def add_differential_privacy_noise(
        self,
        model_params: Mapping[str, torch.Tensor],
        sensitivity: float = 1.0,
        clip_norm: Optional[float] = None,
    ) -> OrderedDict:
        """Backward-compatible alias for publication-level DP only.

        Do NOT call this on an individual client update. Client-update DP is
        implemented by ``EfficientDPManager.add_dp_noise``.
        """
        del clip_norm
        protected, _ = self.add_publication_dp_noise(model_params, sensitivity=sensitivity)
        return protected

    def privacy_report(self, total_rounds: Optional[int] = None) -> Dict[str, Any]:
        report = self.publication_dp.privacy_report()
        report["scope"] = "server_publication"
        if total_rounds is not None:
            report["requested_total_rounds"] = int(total_rounds)
        return report

    # ------------------------------------------------------------------
    # IPFS
    # ------------------------------------------------------------------

    def upload_to_ipfs(self, data: bytes, pin: bool = True) -> str:
        if self.ipfs is None:
            raise RuntimeError("IPFS is not connected.")
        start = time.time()
        cid = self.ipfs.add_bytes(data)
        if pin:
            self.ipfs.pin.add(cid)
        self.performance_metrics["upload_times"].append(time.time() - start)
        return str(cid)

    def download_from_ipfs(self, ipfs_hash: str, timeout: int = 60) -> bytes:
        if self.ipfs is None:
            raise RuntimeError("IPFS is not connected.")
        start = time.time()
        data = self.ipfs.cat(ipfs_hash, timeout=timeout)
        self.performance_metrics["download_times"].append(time.time() - start)
        return bytes(data)

    # ------------------------------------------------------------------
    # Blockchain governance
    # ------------------------------------------------------------------

    def register_client_on_chain(
        self,
        client_address: str,
        client_private_key: Optional[str],
        public_key_pem: bytes,
        stake_amount: float = 0.01,
    ) -> str:
        if self.contract is None:
            raise RuntimeError("Contract not initialized.")
        sender = Web3.to_checksum_address(client_address)
        value = self.w3.to_wei(float(stake_amount), "ether")
        start = time.time()
        tx = self._send_function(
            self.contract.functions.registerClient(self.get_public_key_hash(public_key_pem)),
            sender=sender,
            private_key=client_private_key or None,
            value=value,
            gas=250_000,
        )
        self.performance_metrics["blockchain_times"].append(time.time() - start)
        return tx

    def start_round_on_chain(self, previous_model_ipfs_hash: str = "") -> str:
        if self.contract is None or self.coordinator_account is None:
            raise RuntimeError("Contract or coordinator is not initialized.")
        start = time.time()
        tx = self._send_function(
            self.contract.functions.startRound(previous_model_ipfs_hash),
            sender=self.coordinator_account.address,
            private_key=(
                self.coordinator_account.key.hex()
                if hasattr(self.coordinator_account, "key")
                else None
            ),
            gas=300_000,
        )
        self.performance_metrics["blockchain_times"].append(time.time() - start)
        return tx

    def submit_local_update_on_chain(
        self,
        client_address: str,
        client_private_key: Optional[str],
        ipfs_hash: str,
        update_hash: bytes,
        data_size: int,
        encrypted_metrics: bytes,
    ) -> str:
        """Record a contribution using the contract's client entry point."""
        if self.contract is None:
            raise RuntimeError("Contract not initialized.")
        sender = Web3.to_checksum_address(client_address)

        # Prefer the new explicit recordContribution coordinator interface for
        # the canonical two-phase runner. This client-sender method remains as a
        # compatibility path when the deployed contract exposes submitLocalUpdate.
        names = set()
        abi = getattr(self.contract, "abi", [])
        names = {item.get("name") for item in abi if item.get("type") == "function"}
        if "submitLocalUpdate" not in names:
            raise RuntimeError(
                "This method requires the compatibility submitLocalUpdate entry point. "
                "Use record_contribution_on_chain() with the corrected contract."
            )

        return self._send_function(
            self.contract.functions.submitLocalUpdate(
                ipfs_hash,
                bytes(update_hash),
                int(data_size),
                bytes(encrypted_metrics),
            ),
            sender=sender,
            private_key=client_private_key or None,
            gas=350_000,
        )

    def record_contributions_batch_on_chain(
        self,
        client_addresses: Sequence[str],
        round_number: int,
        update_hashes: Sequence[bytes],
        data_sizes: Sequence[int],
    ) -> str:
        """Record all round contributions in one coordinator transaction."""
        if self.contract is None or self.coordinator_account is None:
            raise RuntimeError("Contract or coordinator is not initialized.")
        addresses = [Web3.to_checksum_address(a) for a in client_addresses]
        hashes = [bytes(h) for h in update_hashes]
        sizes = [int(x) for x in data_sizes]
        if not addresses or len(addresses) != len(hashes) or len(addresses) != len(sizes):
            raise ValueError("Batch contribution arrays must be non-empty and equal length.")

        abi_names = {
            item.get("name") for item in getattr(self.contract, "abi", [])
            if item.get("type") == "function"
        }
        if "recordContributionsBatch" not in abi_names:
            raise RuntimeError("Deployed contract does not expose recordContributionsBatch().")

        return self._send_function(
            self.contract.functions.recordContributionsBatch(addresses, hashes, sizes),
            sender=self.coordinator_account.address,
            private_key=(
                self.coordinator_account.key.hex()
                if hasattr(self.coordinator_account, "key")
                else None
            ),
            gas=max(450_000, 180_000 * len(addresses)),
        )

    def update_reputations_batch_on_chain(
        self,
        client_addresses: Sequence[str],
        reputation_signals: Sequence[int | float],
    ) -> str:
        """Apply the on-chain leaky-integrator reputation recurrence in one tx."""
        if self.contract is None or self.coordinator_account is None:
            raise RuntimeError("Contract or coordinator is not initialized.")

        addresses = [Web3.to_checksum_address(a) for a in client_addresses]
        signals = [int(round(float(x))) for x in reputation_signals]
        if not addresses or len(addresses) != len(signals):
            raise ValueError("Batch reputation arrays must be non-empty and equal length.")
        if any(x < 0 or x > 1000 for x in signals):
            raise ValueError("Reputation signals must be in the 0..1000 range.")

        abi_names = {
            item.get("name") for item in getattr(self.contract, "abi", [])
            if item.get("type") == "function"
        }
        if "updateReputationsBatch" not in abi_names:
            raise RuntimeError("Deployed contract does not expose updateReputationsBatch().")

        return self._send_function(
            self.contract.functions.updateReputationsBatch(addresses, signals),
            sender=self.coordinator_account.address,
            private_key=(
                self.coordinator_account.key.hex()
                if hasattr(self.coordinator_account, "key")
                else None
            ),
            gas=max(450_000, 150_000 * len(addresses)),
        )

    def is_client_registered(self, client_address: str) -> bool:
        """Return registration state using the contract's public clients mapping."""
        if self.contract is None:
            raise RuntimeError("Contract not initialized.")
        address = Web3.to_checksum_address(client_address)
        abi_names = {
            item.get("name") for item in getattr(self.contract, "abi", [])
            if item.get("type") == "function"
        }
        if "clients" not in abi_names:
            return False
        data = self.contract.functions.clients(address).call()
        return bool(data[0])

    def record_contribution_on_chain(
        self,
        client_address: str,
        round_number: int,
        update_hash: bytes,
        data_size: int,
    ) -> str:
        """Record lightweight contribution metadata for the current round."""
        if self.contract is None or self.coordinator_account is None:
            raise RuntimeError("Contract or coordinator is not initialized.")
        return self._send_function(
            self.contract.functions.recordContribution(
                Web3.to_checksum_address(client_address),
                int(round_number),
                bytes(update_hash),
                int(data_size),
            ),
            sender=self.coordinator_account.address,
            private_key=(
                self.coordinator_account.key.hex()
                if hasattr(self.coordinator_account, "key")
                else None
            ),
            gas=350_000,
        )

    def update_reputation_on_chain(
        self,
        client_address: str,
        reputation_signal: int | float,
        round_number: int,
    ) -> str:
        """Apply one bounded reputation target using Eq. (4.4.4)."""
        if self.contract is None or self.coordinator_account is None:
            raise RuntimeError("Contract or coordinator is not initialized.")
        signal = int(round(float(reputation_signal)))
        if signal < 0 or signal > 1000:
            raise ValueError("Reputation signal must be in the 0..1000 range.")
        return self._send_function(
            self.contract.functions.updateReputation(
                Web3.to_checksum_address(client_address),
                int(round_number),
                signal,
            ),
            sender=self.coordinator_account.address,
            private_key=(
                self.coordinator_account.key.hex()
                if hasattr(self.coordinator_account, "key")
                else None
            ),
            gas=300_000,
        )

    def verify_local_update_on_chain(
        self,
        client_address: str,
        round_number: int,
        valid: bool,
    ) -> str:
        """Legacy boolean adapter; canonical BC-AWFedAvg uses reputation targets."""
        if self.contract is None or self.coordinator_account is None:
            raise RuntimeError("Contract or coordinator is not initialized.")
        return self._send_function(
            self.contract.functions.verifyLocalUpdate(
                Web3.to_checksum_address(client_address),
                int(round_number),
                bool(valid),
            ),
            sender=self.coordinator_account.address,
            private_key=(
                self.coordinator_account.key.hex()
                if hasattr(self.coordinator_account, "key")
                else None
            ),
            gas=300_000,
        )

    def submit_aggregated_model_on_chain(
        self,
        round_number: int,
        ipfs_hash: str,
        model_hash: bytes,
    ) -> str:
        if self.contract is None or self.coordinator_account is None:
            raise RuntimeError("Contract or coordinator is not initialized.")
        return self._send_function(
            self.contract.functions.submitAggregatedModel(
                int(round_number),
                str(ipfs_hash),
                bytes(model_hash),
            ),
            sender=self.coordinator_account.address,
            private_key=(
                self.coordinator_account.key.hex()
                if hasattr(self.coordinator_account, "key")
                else None
            ),
            gas=350_000,
        )

    # ------------------------------------------------------------------
    # Blockchain views
    # ------------------------------------------------------------------

    def get_model_from_chain(self, round_number: int) -> Dict[str, Any]:
        if self.contract is None:
            raise RuntimeError("Contract not initialized.")
        item = self.contract.functions.getModelInfo(int(round_number)).call()
        return {
            "ipfs_hash": item[0],
            "model_hash": item[1],
            "timestamp": item[2],
            "num_contributors": item[3],
            "is_aggregated": item[4],
        }

    def get_client_info(self, client_address: str) -> Dict[str, Any]:
        if self.contract is None:
            raise RuntimeError("Contract not initialized.")
        item = self.contract.functions.getClientInfo(
            Web3.to_checksum_address(client_address)
        ).call()
        return {
            "is_active": bool(item[0]),
            "reputation": int(item[1]),
            "total_contributions": int(item[2]),
            "staked_amount": int(item[3]),
        }

    # ------------------------------------------------------------------
    # Legacy workflow guards
    # ------------------------------------------------------------------

    def client_upload_model(self, *args, **kwargs):
        """Deprecated compatibility wrapper for archive-only publication.

        It intentionally does not apply client-update DP. Using this function
        for the learning contribution would bypass the canonical DP -> weight
        -> SecAgg protocol.
        """
        if not args and "model_params" not in kwargs:
            raise ValueError("model_params is required.")
        model_params = kwargs.get("model_params", args[1] if len(args) > 1 else None)
        if model_params is None:
            raise ValueError("model_params is required.")
        compression = bool(kwargs.get("compression", True))
        data_size = int(kwargs.get("data_size", args[2] if len(args) > 2 else 0))
        metrics = kwargs.get("metrics", args[3] if len(args) > 3 else {})

        encrypted, key = self.encrypt_model(model_params, compression=compression)
        cid = self.upload_to_ipfs(encrypted, pin=True)
        update_hash = hashlib.sha256(encrypted).digest()
        encrypted_metrics = AESGCM(key).encrypt(
            os.urandom(AES_NONCE_SIZE),
            json.dumps(metrics, sort_keys=True).encode("utf-8"),
            AES_VERSION,
        )
        del data_size
        return cid, update_hash, encrypted_metrics

    def coordinator_download_and_aggregate(self, *args, **kwargs):
        raise RuntimeError(
            "Disabled: BC-AWFedAvg Secure Aggregation does not permit downloading and "
            "decrypting individual client contributions at the coordinator. "
            "Use the two-phase secure-aggregation runner."
        )

    def print_performance_summary(self) -> None:
        print("\n" + "=" * 70)
        print("PRIVACY / IPFS / BLOCKCHAIN BACKEND")
        print("=" * 70)
        for name, values in self.performance_metrics.items():
            if not values:
                continue
            if all(isinstance(v, (int, float)) for v in values):
                print(f"  {name:24s}: mean={float(np.mean(values)):.6f}s")
            else:
                print(f"  {name:24s}: {len(values)} records")
        report = self.privacy_report()
        print(f"  publication RDP eps      : {report.get('eps_rdp', 0.0):.6f}")
        print(f"  publication rounds       : {report.get('rounds', 0)}")


__all__ = ["PrivacyPreservingFederatedLearning"]
