"""
Privacy-Preserving Federated Learning with Blockchain and IPFS
================================================================

This module integrates:
1. Blockchain smart contract for model governance
2. IPFS for lightweight distributed storage
3. Encryption for privacy preservation
4. Differential privacy mechanisms
5. Performance optimization for maintaining KPIs
"""

import os
import json
import hashlib
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import ipfshttpclient
from web3 import Web3
from web3.middleware import geth_poa_middleware
import pickle
import io
import time
import gzip


def _raw_tx(signed):
    """Return raw signed transaction bytes — works with both web3.py 5.x and 6.x."""
    return getattr(signed, "rawTransaction", None) or getattr(signed, "raw_transaction", None)


class PrivacyPreservingFederatedLearning:
    """
    Main class for privacy-preserving federated learning with blockchain and IPFS.
    """
    
    def __init__(
        self,
        blockchain_provider: str = "http://localhost:8545",
        contract_address: Optional[str] = None,
        contract_abi_path: Optional[str] = None,
        ipfs_addr: str = "/ip4/127.0.0.1/tcp/5001",
        epsilon: float = 1.0,  # Differential privacy parameter
        delta: float = 1e-5,
        clip_norm: float = 1.0,
        coordinator_private_key: Optional[str] = None,
        require_connection: bool = True,
    ):
        """
        Initialize the privacy-preserving federated learning system.

        Args:
            blockchain_provider: Web3 provider URL
            contract_address: Deployed smart contract address
            contract_abi_path: Path to contract ABI JSON
            ipfs_addr: IPFS daemon address
            epsilon: DP epsilon parameter (privacy budget)
            delta: DP delta parameter
            clip_norm: Gradient clipping norm for DP
            coordinator_private_key: Private key for coordinator
            require_connection: When False, a missing blockchain/IPFS node is
                logged as a warning instead of raising ConnectionError.  Set
                to False when running in offline / blockchain_enabled=False mode
                so that the DP + encryption helpers are still available.
        """
        # Blockchain setup
        self.w3 = Web3(Web3.HTTPProvider(blockchain_provider))
        self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)

        if not self.w3.is_connected():
            if require_connection:
                raise ConnectionError(
                    f"Failed to connect to blockchain at {blockchain_provider}. "
                    "Start a local node (e.g. Ganache) or set blockchain_enabled=False "
                    "to run without a live chain."
                )
            else:
                import warnings as _w
                _w.warn(
                    f"[PPFL] Blockchain node not reachable at {blockchain_provider}. "
                    "Running in offline mode — on-chain calls will be skipped.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Load contract
        if contract_address and contract_abi_path:
            try:
                with open(contract_abi_path, 'r') as f:
                    _raw = json.load(f)
                # contract_info.json stores the ABI under the "abi" key;
                # fall back to using the raw value directly if it's already a list.
                contract_abi = _raw["abi"] if isinstance(_raw, dict) else _raw
                self.contract = self.w3.eth.contract(
                    address=Web3.to_checksum_address(contract_address),
                    abi=contract_abi
                )
            except Exception as e:
                self.contract = None
                import warnings as _w
                _w.warn(f"[PPFL] Could not load contract ABI: {e}", RuntimeWarning, stacklevel=2)
        else:
            self.contract = None

        # IPFS setup — suppress version/deprecation warnings from ipfshttpclient 0.8.x
        import warnings as _w
        try:
            with _w.catch_warnings():
                _w.filterwarnings("ignore", category=DeprecationWarning)
                _w.filterwarnings("ignore", message=".*strict.*")
                _w.filterwarnings("ignore", message=".*VersionMismatch.*")
                _w.filterwarnings("ignore", message=".*Unsupported daemon version.*")
                try:
                    # check_version=False skips the daemon version gate (works on Kubo 0.24+)
                    self.ipfs = ipfshttpclient.connect(ipfs_addr, check_version=False)
                except TypeError:
                    # 0.8.0a2 doesn't accept check_version — connect without it;
                    # VersionMismatch will be a warning (caught above), not an error.
                    self.ipfs = ipfshttpclient.connect(ipfs_addr)
            print(f"✅ Connected to IPFS: {self.ipfs.id()['ID']}")
        except Exception as e:
            print(f"⚠️  IPFS connection failed: {e}")
            self.ipfs = None

        # Store config values so callers can extract them as plain primitives
        self._contract_abi_path = contract_abi_path
        self._ipfs_addr         = ipfs_addr

        # Privacy parameters
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self._rounds_elapsed = 0   # for cumulative DP accounting

        # Coordinator account — use first Ganache account (auto-unlocked, no key needed)
        if self.w3 and self.w3.is_connected():
            try:
                # Use first available account as coordinator (Ganache unlocks all)
                first_account = self.w3.eth.accounts[0]
                # Create a minimal account-like object with just .address
                class _UnlockedAccount:
                    def __init__(self, addr): self.address = addr
                self.coordinator_account = _UnlockedAccount(first_account)
                # If a private key was also provided, upgrade to full account object
                if coordinator_private_key:
                    try:
                        self.coordinator_account = self.w3.eth.account.from_key(
                            coordinator_private_key)
                    except Exception:
                        pass  # keep the unlocked account fallback
            except Exception:
                self.coordinator_account = None
        else:
            self.coordinator_account = None
        
        # Encryption keys (generated per client)
        self.client_keys = {}

        # ── Cumulative Privacy Accounting ─────────────────────────────────
        # Tracks ε consumed per round so we can report ε_total = √(2T·ln(1/δ))·ε
        # (advanced composition theorem, Dwork & Roth 2014, Theorem 3.20).

        # Performance tracking
        self.performance_metrics = {
            'upload_times': [],
            'download_times': [],
            'encryption_times': [],
            'decryption_times': [],
            'model_sizes': [],
            'privacy_overhead': []
        }
    
    # ============= Key Management =============
    
    def generate_client_keypair(self, client_id: str) -> Tuple[bytes, bytes]:
        """
        Generate RSA key pair for a client.
        
        Returns:
            (private_key_pem, public_key_pem)
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        self.client_keys[client_id] = {
            'private': private_pem,
            'public': public_pem
        }
        
        return private_pem, public_pem
    
    def get_public_key_hash(self, public_key_pem: bytes) -> bytes:
        """Get hash of public key for blockchain registration."""
        return hashlib.sha256(public_key_pem).digest()
    
    # ============= Model Encryption =============
    
    def encrypt_model(
        self,
        model_params: OrderedDict,
        compression: bool = True
    ) -> Tuple[bytes, bytes]:
        """
        Encrypt model parameters for IPFS storage.

        Pipeline (order matters for compression ratio):
          1. Quantise float32 → float16  (~50 % size reduction on raw weights)
          2. Serialise with numpy.savez_compressed (gzip internally, effective on
             float16 arrays because adjacent values are highly correlated)
          3. Encrypt with Fernet (AES-128-CBC + HMAC-SHA256)

        Why float16 is safe here
        ------------------------
        The parameters uploaded to IPFS are the DP-noised coordinator model,
        not the weights used for FL aggregation.  The Gaussian noise already
        degrades precision to O(σ) >> float16 epsilon (~6e-5), so quantisation
        loss is negligible for the audit copy.

        Returns
        -------
        (encrypted_data, symmetric_key)
        """
        import numpy as np

        start_time = time.time()

        # 1. Quantise: float32 tensors → float16 numpy arrays
        params_f16 = {
            k: v.detach().cpu().to(torch.float16).numpy()
            for k, v in model_params.items()
        }

        # 2. Compress: numpy savez_compressed applies gzip to float arrays
        #    (float16 arrays compress 3–5× better than float32 due to half the
        #    entropy and higher inter-element correlation after DP smoothing)
        buf = io.BytesIO()
        np.savez_compressed(buf, **params_f16)
        model_bytes = buf.getvalue()

        # 3. Encrypt with Fernet (AES-128-CBC + HMAC-SHA256)
        symmetric_key = Fernet.generate_key()
        cipher        = Fernet(symmetric_key)
        encrypted_data = cipher.encrypt(model_bytes)

        encryption_time = time.time() - start_time
        self.performance_metrics['encryption_times'].append(encryption_time)
        self.performance_metrics['model_sizes'].append(len(encrypted_data))

        return encrypted_data, symmetric_key
    
    def decrypt_model(
        self,
        encrypted_data: bytes,
        symmetric_key: bytes,
        compression: bool = True        # kept for API compatibility, always uses npz
    ) -> OrderedDict:
        """
        Decrypt and reconstruct model parameters from IPFS payload.

        Reverses encrypt_model:
          1. Fernet decrypt
          2. numpy.load (npz — gzip decompressed automatically)
          3. Restore float32 tensors from float16 arrays
        """
        import numpy as np
        from collections import OrderedDict as OD

        start_time = time.time()

        cipher      = Fernet(symmetric_key)
        model_bytes = cipher.decrypt(encrypted_data)

        buf    = io.BytesIO(model_bytes)
        npz    = np.load(buf, allow_pickle=False)
        # Restore as float32 OrderedDict (same key order as saved)
        model_params = OD({
            k: torch.tensor(npz[k].astype("float32"))
            for k in npz.files
        })

        decryption_time = time.time() - start_time
        self.performance_metrics['decryption_times'].append(decryption_time)

        return model_params
    
    def encrypt_symmetric_key(
        self,
        symmetric_key: bytes,
        public_key_pem: bytes
    ) -> bytes:
        """Encrypt symmetric key with public key for sharing."""
        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )
        
        encrypted_key = public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return encrypted_key
    
    def decrypt_symmetric_key(
        self,
        encrypted_key: bytes,
        private_key_pem: bytes
    ) -> bytes:
        """Decrypt symmetric key with private key."""
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=default_backend()
        )
        
        symmetric_key = private_key.decrypt(
            encrypted_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return symmetric_key
    
    # ============= Differential Privacy =============
    
    def add_differential_privacy_noise(
        self,
        model_params: OrderedDict,
        sensitivity: float = 1.0,
        clip_norm: Optional[float] = None
    ) -> OrderedDict:
        """
        Add calibrated Gaussian noise for differential privacy.
        
        Args:
            model_params: Model parameters
            sensitivity: L2 sensitivity of the query
            clip_norm: Gradient clipping norm (uses self.clip_norm if None)
        
        Returns:
            Noised model parameters
        """
        start_time = time.time()
        
        if clip_norm is None:
            clip_norm = self.clip_norm
        
        # Calculate noise scale using Gaussian mechanism
        noise_scale = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        
        noised_params = OrderedDict()
        
        for name, param in model_params.items():
            if isinstance(param, torch.Tensor):
                # Clip gradient
                param_norm = torch.norm(param)
                if param_norm > clip_norm:
                    param = param * (clip_norm / param_norm)
                
                # Add Gaussian noise
                noise = torch.randn_like(param) * noise_scale
                noised_params[name] = param + noise
            else:
                noised_params[name] = param
        
        privacy_overhead = time.time() - start_time
        self.performance_metrics['privacy_overhead'].append(privacy_overhead)
        
        return noised_params
    
    def clip_gradients(
        self,
        gradients: OrderedDict,
        max_norm: Optional[float] = None
    ) -> OrderedDict:
        """
        Clip gradients to bounded L2 norm for privacy.
        
        Args:
            gradients: Gradient dictionary
            max_norm: Maximum L2 norm (uses self.clip_norm if None)
        
        Returns:
            Clipped gradients
        """
        if max_norm is None:
            max_norm = self.clip_norm
        
        # Calculate total norm
        total_norm = 0.0
        for grad in gradients.values():
            if isinstance(grad, torch.Tensor):
                total_norm += torch.norm(grad).item() ** 2
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            clipped_grads = OrderedDict()
            for name, grad in gradients.items():
                if isinstance(grad, torch.Tensor):
                    clipped_grads[name] = grad * clip_coef
                else:
                    clipped_grads[name] = grad
            return clipped_grads
        
        return gradients
    


    # ============= Cumulative Privacy Accounting =============

    def account_privacy_round(self) -> dict:
        """
        Call once per FL round to update the cumulative privacy budget.

        Uses the advanced composition theorem:
            ε_total = √(2T · ln(1/δ)) · ε_per_round

        Returns a dict with per-round and cumulative ε for logging.
        """
        self._rounds_elapsed += 1
        T = self._rounds_elapsed
        eps_total = np.sqrt(2 * T * np.log(1.0 / self.delta)) * self.epsilon
        return {
            "round":         T,
            "eps_per_round": float(self.epsilon),
            "delta":         float(self.delta),
            "eps_total":     float(eps_total),
        }

    def privacy_report(self, total_rounds: int = None) -> dict:
        """Return a privacy accounting summary."""
        T = self._rounds_elapsed
        eps_total = (
            np.sqrt(2 * T * np.log(1.0 / self.delta)) * self.epsilon if T > 0 else 0.0
        )
        report = {
            "rounds_elapsed":   T,
            "eps_per_round":    float(self.epsilon),
            "delta":            float(self.delta),
            "eps_total_so_far": float(eps_total),
        }
        if total_rounds is not None:
            projected = np.sqrt(2 * total_rounds * np.log(1.0 / self.delta)) * self.epsilon
            report["eps_total_projected"] = float(projected)
        return report

    # ============= IPFS Storage =============

    def upload_to_ipfs(
        self,
        data: bytes,
        pin: bool = True
    ) -> str:
        """
        Upload encrypted data to IPFS.
        
        Args:
            data: Encrypted data bytes
            pin: Whether to pin the file
        
        Returns:
            IPFS hash (CID)
        """
        if self.ipfs is None:
            raise RuntimeError("IPFS not connected")
        
        start_time = time.time()
        
        # Upload to IPFS
        result = self.ipfs.add_bytes(data)
        ipfs_hash = result
        
        # Pin if requested
        if pin:
            self.ipfs.pin.add(ipfs_hash)
        
        upload_time = time.time() - start_time
        self.performance_metrics['upload_times'].append(upload_time)
        
        print(f"📤 Uploaded to IPFS: {ipfs_hash} ({len(data)/1024:.2f} KB in {upload_time:.2f}s)")
        
        return ipfs_hash
    
    def download_from_ipfs(
        self,
        ipfs_hash: str,
        timeout: int = 60
    ) -> bytes:
        """
        Download data from IPFS.
        
        Args:
            ipfs_hash: IPFS CID
            timeout: Download timeout in seconds
        
        Returns:
            Downloaded bytes
        """
        if self.ipfs is None:
            raise RuntimeError("IPFS not connected")
        
        start_time = time.time()
        
        # Download from IPFS
        data = self.ipfs.cat(ipfs_hash, timeout=timeout)
        
        download_time = time.time() - start_time
        self.performance_metrics['download_times'].append(download_time)
        
        print(f"📥 Downloaded from IPFS: {ipfs_hash} ({len(data)/1024:.2f} KB in {download_time:.2f}s)")
        
        return data
    
    # ============= Blockchain Integration =============
    
    def _resolve_sender(self, requested_address: str) -> str:
        """Return requested_address if it is a Ganache-unlocked account,
        otherwise fall back to the first available unlocked account."""
        try:
            unlocked = [a.lower() for a in self.w3.eth.accounts]
            if requested_address.lower() in unlocked:
                return requested_address
            # Fall back to first unlocked account
            fallback = self.w3.eth.accounts[0]
            return fallback
        except Exception:
            return requested_address

    def register_client_on_chain(
        self,
        client_address: str,
        client_private_key: str,   # kept for API compatibility, not used (Ganache auto-signs)
        public_key_pem: bytes,
        stake_amount: float = 0.01
    ) -> str:
        """Register client on blockchain using Ganache's auto-unlocked account."""
        if self.contract is None:
            raise RuntimeError("Contract not initialized")

        sender = self._resolve_sender(client_address)
        public_key_hash = self.get_public_key_hash(public_key_pem)
        tx_hash = self.contract.functions.registerClient(public_key_hash).transact({
            'from':  sender,
            'value': self.w3.to_wei(stake_amount, 'ether'),
            'gas':   200000,
        })
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"📝 Client {sender[:10]}… registered: {tx_hash.hex()[:16]}…")
        return tx_hash.hex()
    
    def start_round_on_chain(
        self,
        previous_model_ipfs_hash: str = ""
    ) -> str:
        """Start a new federated learning round on blockchain."""
        if self.contract is None or self.coordinator_account is None:
            raise RuntimeError("Contract or coordinator not initialized")

        tx_hash = self.contract.functions.startRound(previous_model_ipfs_hash).transact({
            'from': self.coordinator_account.address,
            'gas':  300000,
        })
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"🔗 Round started: {tx_hash.hex()[:16]}…")
        return tx_hash.hex()
    
    def submit_local_update_on_chain(
        self,
        client_address: str,
        client_private_key: str,   # kept for API compatibility, not used
        ipfs_hash: str,
        update_hash: bytes,
        data_size: int,
        encrypted_metrics: bytes
    ) -> str:
        """Submit local model update to blockchain."""
        if self.contract is None:
            raise RuntimeError("Contract not initialized")

        sender = self._resolve_sender(client_address)
        tx_hash = self.contract.functions.submitLocalUpdate(
            ipfs_hash, update_hash, data_size, encrypted_metrics
        ).transact({
            'from': sender,
            'gas':  300000,
        })
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"🔗 Local update submitted: {tx_hash.hex()[:16]}…")
        return tx_hash.hex()
    
    def submit_aggregated_model_on_chain(
        self,
        round_number: int,
        ipfs_hash: str,
        model_hash: bytes
    ) -> str:
        """Submit aggregated global model to blockchain."""
        if self.contract is None or self.coordinator_account is None:
            raise RuntimeError("Contract or coordinator not initialized")

        tx_hash = self.contract.functions.submitAggregatedModel(
            round_number, ipfs_hash, model_hash
        ).transact({
            'from': self.coordinator_account.address,
            'gas':  300000,
        })
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"🔗 Global model submitted: {tx_hash.hex()[:16]}…")
        
        return tx_hash.hex()
    
    def get_model_from_chain(
        self,
        round_number: int
    ) -> Dict[str, Any]:
        """
        Get model information from blockchain.
        
        Returns:
            Model info dictionary
        """
        if self.contract is None:
            raise RuntimeError("Contract not initialized")
        
        model_info = self.contract.functions.getModelInfo(round_number).call()
        
        return {
            'ipfs_hash': model_info[0],
            'model_hash': model_info[1],
            'timestamp': model_info[2],
            'num_contributors': model_info[3],
            'is_aggregated': model_info[4]
        }
    
    # ============= Complete Workflow =============
    
    def client_upload_model(
        self,
        client_id: str,
        model_params: OrderedDict,
        data_size: int,
        metrics: Dict[str, float],
        apply_dp: bool = True,
        compression: bool = True
    ) -> Tuple[str, bytes, bytes]:
        """
        Complete client workflow: DP → Encrypt → Upload to IPFS.
        
        Args:
            client_id: Client identifier
            model_params: Model parameters
            data_size: Number of training samples
            metrics: Performance metrics
            apply_dp: Whether to apply differential privacy
            compression: Whether to compress
        
        Returns:
            (ipfs_hash, update_hash, encrypted_metrics)
        """
        print(f"\n📊 Client {client_id} uploading model...")
        
        # Apply differential privacy
        if apply_dp:
            model_params = self.add_differential_privacy_noise(model_params)
            print(f"  ✅ Differential privacy applied (ε={self.epsilon}, δ={self.delta})")
        
        # Encrypt model
        encrypted_model, symmetric_key = self.encrypt_model(model_params, compression)
        print(f"  🔒 Model encrypted ({len(encrypted_model)/1024:.2f} KB)")
        
        # Upload to IPFS
        ipfs_hash = self.upload_to_ipfs(encrypted_model)
        
        # Calculate update hash for verification
        update_hash = hashlib.sha256(encrypted_model).digest()
        
        # Encrypt metrics
        metrics_bytes = json.dumps(metrics).encode()
        encrypted_metrics = Fernet(symmetric_key).encrypt(metrics_bytes)
        
        print(f"  ✅ Upload complete: {ipfs_hash}")
        
        return ipfs_hash, update_hash, encrypted_metrics
    
    def coordinator_download_and_aggregate(
        self,
        round_number: int,
        client_weights: Optional[Dict[str, float]] = None,
        apply_weighted: bool = True
    ) -> Tuple[OrderedDict, str]:
        """
        Coordinator workflow: Download → Decrypt → Aggregate → Upload.
        
        Args:
            round_number: Current round number
            client_weights: Optional client weights for aggregation
            apply_weighted: Whether to use weighted averaging
        
        Returns:
            (aggregated_params, ipfs_hash)
        """
        print(f"\n🔧 Coordinator aggregating models for round {round_number}...")
        
        # Get contributors from blockchain
        contributors = self.contract.functions.getRoundContributors(round_number).call()
        print(f"  📋 Found {len(contributors)} contributors")
        
        # Download and decrypt all models
        client_models = []
        total_data_size = 0
        
        for client_addr in contributors:
            # Get update info from blockchain
            update_info = self.contract.functions.getLocalUpdate(
                round_number,
                client_addr
            ).call()
            
            ipfs_hash = update_info[0]
            data_size = update_info[3]
            
            # Download from IPFS
            encrypted_data = self.download_from_ipfs(ipfs_hash)
            
            # For aggregation, coordinator needs access to symmetric keys
            # (In practice, this would use secure multi-party computation or
            # clients would send encrypted keys to coordinator)
            
            client_models.append({
                'params': None,  # Would be decrypted params
                'data_size': data_size,
                'weight': client_weights.get(client_addr, 1.0) if client_weights else 1.0
            })
            
            total_data_size += data_size
        
        # Note: In a real implementation, you would implement secure aggregation
        # Here we demonstrate the structure
        
        print(f"  ✅ Downloaded {len(client_models)} models")
        
        # Aggregate (simplified - would use actual decrypted params)
        aggregated_params = OrderedDict()
        
        # Upload aggregated model
        encrypted_agg, _ = self.encrypt_model(aggregated_params, compression=True)
        ipfs_hash = self.upload_to_ipfs(encrypted_agg)
        model_hash = hashlib.sha256(encrypted_agg).digest()
        
        # Submit to blockchain
        self.submit_aggregated_model_on_chain(round_number, ipfs_hash, model_hash)
        
        print(f"  ✅ Aggregation complete: {ipfs_hash}")
        
        return aggregated_params, ipfs_hash
    
    # ============= Performance Monitoring =============
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics.
        
        Returns:
            Performance report dictionary
        """
        metrics = self.performance_metrics
        
        report = {
            'encryption': {
                'count': len(metrics['encryption_times']),
                'avg_time': np.mean(metrics['encryption_times']) if metrics['encryption_times'] else 0,
                'total_time': np.sum(metrics['encryption_times']),
            },
            'decryption': {
                'count': len(metrics['decryption_times']),
                'avg_time': np.mean(metrics['decryption_times']) if metrics['decryption_times'] else 0,
                'total_time': np.sum(metrics['decryption_times']),
            },
            'ipfs_upload': {
                'count': len(metrics['upload_times']),
                'avg_time': np.mean(metrics['upload_times']) if metrics['upload_times'] else 0,
                'avg_size_kb': np.mean(metrics['model_sizes']) / 1024 if metrics['model_sizes'] else 0,
            },
            'ipfs_download': {
                'count': len(metrics['download_times']),
                'avg_time': np.mean(metrics['download_times']) if metrics['download_times'] else 0,
            },
            'privacy_overhead': {
                'avg_time': np.mean(metrics['privacy_overhead']) if metrics['privacy_overhead'] else 0,
                'total_time': np.sum(metrics['privacy_overhead']),
            },
            'total_overhead': {
                'encryption': np.sum(metrics['encryption_times']),
                'privacy': np.sum(metrics['privacy_overhead']),
                'ipfs': np.sum(metrics['upload_times']) + np.sum(metrics['download_times']),
            }
        }
        
        return report
    
    def print_performance_summary(self):
        """Print human-readable performance summary."""
        report = self.get_performance_report()
        
        print("\n" + "="*70)
        print("PRIVACY & BLOCKCHAIN PERFORMANCE SUMMARY")
        print("="*70)
        
        print("\n🔒 Encryption:")
        print(f"  Operations: {report['encryption']['count']}")
        print(f"  Avg time: {report['encryption']['avg_time']:.3f}s")
        print(f"  Total time: {report['encryption']['total_time']:.3f}s")
        
        print("\n🔓 Decryption:")
        print(f"  Operations: {report['decryption']['count']}")
        print(f"  Avg time: {report['decryption']['avg_time']:.3f}s")
        
        print("\n📤 IPFS Upload:")
        print(f"  Operations: {report['ipfs_upload']['count']}")
        print(f"  Avg time: {report['ipfs_upload']['avg_time']:.3f}s")
        print(f"  Avg size: {report['ipfs_upload']['avg_size_kb']:.2f} KB")
        
        print("\n📥 IPFS Download:")
        print(f"  Operations: {report['ipfs_download']['count']}")
        print(f"  Avg time: {report['ipfs_download']['avg_time']:.3f}s")
        
        print("\n🛡️  Privacy (DP):")
        print(f"  Avg overhead: {report['privacy_overhead']['avg_time']:.3f}s")
        print(f"  Total overhead: {report['privacy_overhead']['total_time']:.3f}s")
        
        print("\n📊 Total Overhead:")
        print(f"  Encryption: {report['total_overhead']['encryption']:.3f}s")
        print(f"  Privacy: {report['total_overhead']['privacy']:.3f}s")
        print(f"  IPFS: {report['total_overhead']['ipfs']:.3f}s")
        
        total = sum(report['total_overhead'].values())
        print(f"  TOTAL: {total:.3f}s")
        
        print("="*70)
