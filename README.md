BC-AWFedAvg corrected canonical files
====================================

Core protocol files
-------------------
blockchain_awfedavg.py
  - canonical two-phase BC-AWFedAvg orchestrator
  - five-criterion adaptive weighting
  - persistent reputation-aware weighting
  - client-update DP before weighting
  - weight-before-mask secure aggregation
  - gradual isolation/TTI tracking at theta_iso = 1/(2K)

secure_aggregation.py
  - independent persistent pairwise secrets
  - additive pairwise-cancelling masks
  - server-side sum only
  - explicit research-prototype scope limitations

efficient_dp.py
  - client/update Gaussian DP
  - RDP accounting
  - optional Top-K sparsification

privacy_blockchain_fl.py
  - Ethereum/Ganache governance interface
  - contribution metadata
  - on-chain reputation-target submission
  - AES-256-GCM + RSA-4096 off-chain encryption
  - IPFS publication
  - separate publication-level DP

deploy.py
  - solcx/Web3 deployment
  - writes contract_info.json without private keys
  - records BC-AWFedAvg reputation/isolation configuration

contracts/FederatedLearningContract.sol
  - metadata-only governance contract
  - reputation recurrence: rho_t = 0.85 rho_(t-1) + 0.15 g_t
  - reputation represented on a 0..1000 scale
  - initial reputation = 1000/K
  - no hard minimum-reputation participation gate
  - isolation is measured from aggregation weight, not client deactivation

Important thesis-consistency note
---------------------------------
The thesis version that explicitly specifies the reputation dynamics uses a
leaky-integrator recurrence with beta=0.85. It describes g(.) as a bounded
function of E/R/S signals but does not provide a unique closed-form g(.).
The canonical runner therefore uses the arithmetic mean of the normalized
E/R/S scores as a reproducible operationalization.

The latest thesis PDF version available in the file set describes the reputation
mechanism qualitatively as gradual and defines theta_iso = 1/(2K), but does not
repeat the beta=0.85 recurrence. The explicit beta=0.85 implementation above
therefore follows the thesis version that actually defines the recurrence.

Reputation / TTI synchronization
--------------------------------
The canonical experiment runner explicitly documents and records:
  - s_rep,k^(t) = rho_k^(t) / sum_j rho_j^(t)
  - rho_k^(t) = beta*rho_k^(t-1) + (1-beta)*g(E_k^(t),R_k^(t),S_k^(t))
  - beta = 0.85
  - w_k^(0) = 1/K
  - five criteria alpha = (0.22, 0.38, 0.20, 0.15, 0.05)
  - eta = 0.7 exponential weight smoothing
  - theta_iso = 1/(2K)
  - TTI_k = first round with w_k^(t) < theta_iso
  - reputation is gradual influence control, not binary client exclusion

The current thesis specifies g(.) as a bounded function of E/R/S-related scores,
but does not give a unique closed-form definition of g. The implementation uses
the arithmetic mean of the normalized E/R/S scores as an explicit reproducibility
convention. This distinction is recorded so the repository does not incorrectly
claim that the chosen g(.) expression is copied verbatim from the thesis.

The runner also writes results/thesis_protocol_spec.json so the equations and runtime
parameters used by an experiment are preserved alongside the numerical results.
