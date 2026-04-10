// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FederatedLearningContract
 * @dev Privacy-preserving federated learning with IPFS storage and access control
 * @notice Manages model updates, aggregation verification, and client reputation
 */
contract FederatedLearningContract {
    
    // ==================== State Variables ====================
    
    address public coordinator;
    uint256 public currentRound;
    uint256 public minClientsPerRound;
    uint256 public maxClientsPerRound;
    uint256 public roundTimeout;
    
    // Model versioning and IPFS storage
    struct ModelVersion {
        string ipfsHash;           // IPFS hash of encrypted model
        bytes32 modelHash;         // Hash of model parameters for verification
        uint256 round;
        uint256 timestamp;
        uint256 numContributors;
        bool isAggregated;
        address[] contributors;
        mapping(address => bool) hasContributed;
    }
    
    // Client information and reputation
    struct Client {
        bool isRegistered;
        bool isActive;
        uint256 reputation;        // 0-1000 scale
        uint256 totalContributions;
        uint256 lastContribution;
        bytes32 publicKeyHash;     // Hash of client's public key for encryption
        uint256 stakedAmount;      // Economic incentive for honest participation
    }
    
    // Round-specific update from a client
    struct LocalUpdate {
        string ipfsHash;           // IPFS hash of encrypted local model
        bytes32 updateHash;        // Hash of update for verification
        uint256 timestamp;
        uint256 dataSize;          // Number of samples used
        bytes encryptedMetrics;    // Encrypted performance metrics
        bool isVerified;
    }
    
    // Privacy parameters
    struct PrivacyConfig {
        uint256 minDataSize;       // Minimum samples required
        uint256 noiseMagnitude;    // Differential privacy noise level
        bool requireEncryption;
        uint256 minReputation;     // Minimum reputation to participate
    }
    
    // Storage
    mapping(uint256 => ModelVersion) public globalModels;
    mapping(address => Client) public clients;
    mapping(uint256 => mapping(address => LocalUpdate)) public localUpdates;
    mapping(uint256 => uint256) public roundStartTime;
    
    address[] public registeredClients;
    PrivacyConfig public privacyConfig;
    
    // ==================== Events ====================
    
    event ClientRegistered(address indexed client, bytes32 publicKeyHash);
    event ClientDeactivated(address indexed client, string reason);
    event RoundStarted(uint256 indexed round, uint256 timestamp);
    event LocalUpdateSubmitted(address indexed client, uint256 indexed round, string ipfsHash);
    event ModelAggregated(uint256 indexed round, string ipfsHash, uint256 numContributors);
    event ReputationUpdated(address indexed client, uint256 newReputation);
    event PrivacyConfigUpdated(uint256 minDataSize, uint256 noiseMagnitude);
    event StakeDeposited(address indexed client, uint256 amount);
    event StakeSlashed(address indexed client, uint256 amount, string reason);
    
    // ==================== Modifiers ====================
    
    modifier onlyCoordinator() {
        require(msg.sender == coordinator, "Only coordinator");
        _;
    }
    
    modifier onlyRegisteredClient() {
        require(clients[msg.sender].isRegistered, "Not registered");
        require(clients[msg.sender].isActive, "Client not active");
        _;
    }
    
    modifier validRound(uint256 round) {
        require(round == currentRound, "Invalid round");
        require(block.timestamp <= roundStartTime[round] + roundTimeout, "Round expired");
        _;
    }
    
    // ==================== Constructor ====================
    
    constructor(
        uint256 _minClients,
        uint256 _maxClients,
        uint256 _roundTimeout
    ) {
        coordinator = msg.sender;
        minClientsPerRound = _minClients;
        maxClientsPerRound = _maxClients;
        roundTimeout = _roundTimeout;
        currentRound = 0;
        
        // Default privacy configuration
        privacyConfig = PrivacyConfig({
            minDataSize: 100,
            noiseMagnitude: 1,
            requireEncryption: true,
            minReputation: 500
        });
    }
    
    // ==================== Client Management ====================
    
    /**
     * @dev Register a new client with public key for encryption
     */
    function registerClient(bytes32 _publicKeyHash) external payable {
        require(!clients[msg.sender].isRegistered, "Already registered");
        require(msg.value >= 0.01 ether, "Insufficient stake");
        
        clients[msg.sender] = Client({
            isRegistered: true,
            isActive: true,
            reputation: 500,  // Start with neutral reputation
            totalContributions: 0,
            lastContribution: 0,
            publicKeyHash: _publicKeyHash,
            stakedAmount: msg.value
        });
        
        registeredClients.push(msg.sender);
        
        emit ClientRegistered(msg.sender, _publicKeyHash);
        emit StakeDeposited(msg.sender, msg.value);
    }
    
    /**
     * @dev Deactivate a client (only coordinator or self)
     */
    function deactivateClient(address _client, string calldata _reason) external {
        require(
            msg.sender == coordinator || msg.sender == _client,
            "Unauthorized"
        );
        require(clients[_client].isRegistered, "Not registered");
        
        clients[_client].isActive = false;
        
        emit ClientDeactivated(_client, _reason);
    }
    
    /**
     * @dev Increase stake for better reputation
     */
    function increaseStake() external payable onlyRegisteredClient {
        clients[msg.sender].stakedAmount += msg.value;
        emit StakeDeposited(msg.sender, msg.value);
    }
    
    // ==================== Federated Learning Round ====================
    
    /**
     * @dev Start a new federated learning round
     */
    function startRound(string calldata _previousModelHash) external onlyCoordinator {
        require(
            currentRound == 0 || globalModels[currentRound].isAggregated,
            "Previous round not completed"
        );
        
        currentRound++;
        roundStartTime[currentRound] = block.timestamp;
        
        // Initialize the model version
        ModelVersion storage model = globalModels[currentRound];
        model.round = currentRound;
        model.timestamp = block.timestamp;
        model.ipfsHash = _previousModelHash;
        model.isAggregated = false;
        
        emit RoundStarted(currentRound, block.timestamp);
    }
    
    /**
     * @dev Submit local model update (encrypted and stored on IPFS)
     */
    function submitLocalUpdate(
        string calldata _ipfsHash,
        bytes32 _updateHash,
        uint256 _dataSize,
        bytes calldata _encryptedMetrics
    ) external onlyRegisteredClient validRound(currentRound) {
        require(
            clients[msg.sender].reputation >= privacyConfig.minReputation,
            "Reputation too low"
        );
        require(_dataSize >= privacyConfig.minDataSize, "Insufficient data");
        require(
            !globalModels[currentRound].hasContributed[msg.sender],
            "Already contributed"
        );
        require(
            globalModels[currentRound].numContributors < maxClientsPerRound,
            "Round full"
        );
        
        // Store the local update
        localUpdates[currentRound][msg.sender] = LocalUpdate({
            ipfsHash: _ipfsHash,
            updateHash: _updateHash,
            timestamp: block.timestamp,
            dataSize: _dataSize,
            encryptedMetrics: _encryptedMetrics,
            isVerified: false
        });
        
        // Update global model tracking
        ModelVersion storage model = globalModels[currentRound];
        model.hasContributed[msg.sender] = true;
        model.contributors.push(msg.sender);
        model.numContributors++;
        
        // Update client stats
        clients[msg.sender].totalContributions++;
        clients[msg.sender].lastContribution = block.timestamp;
        
        emit LocalUpdateSubmitted(msg.sender, currentRound, _ipfsHash);
    }
    
    /**
     * @dev Verify a local update (called by coordinator after off-chain verification)
     */
    function verifyLocalUpdate(
        address _client,
        uint256 _round,
        bool _isValid
    ) external onlyCoordinator {
        LocalUpdate storage update = localUpdates[_round][_client];
        require(update.timestamp > 0, "Update not found");
        
        update.isVerified = _isValid;
        
        if (_isValid) {
            // Increase reputation for valid contribution
            uint256 newRep = clients[_client].reputation + 10;
            clients[_client].reputation = newRep > 1000 ? 1000 : newRep;
        } else {
            // Slash stake and decrease reputation for invalid contribution
            uint256 slashAmount = clients[_client].stakedAmount / 10;
            clients[_client].stakedAmount -= slashAmount;
            
            uint256 newRep = clients[_client].reputation > 50 
                ? clients[_client].reputation - 50 
                : 0;
            clients[_client].reputation = newRep;
            
            emit StakeSlashed(_client, slashAmount, "Invalid update");
        }
        
        emit ReputationUpdated(_client, clients[_client].reputation);
    }
    
    /**
     * @dev Submit aggregated global model (encrypted, stored on IPFS)
     */
    function submitAggregatedModel(
        uint256 _round,
        string calldata _ipfsHash,
        bytes32 _modelHash
    ) external onlyCoordinator {
        require(_round == currentRound, "Invalid round");
        require(
            globalModels[_round].numContributors >= minClientsPerRound,
            "Insufficient contributors"
        );
        require(!globalModels[_round].isAggregated, "Already aggregated");
        
        ModelVersion storage model = globalModels[_round];
        model.ipfsHash = _ipfsHash;
        model.modelHash = _modelHash;
        model.isAggregated = true;
        
        emit ModelAggregated(_round, _ipfsHash, model.numContributors);
    }
    
    // ==================== Privacy and Configuration ====================
    
    /**
     * @dev Update privacy configuration
     */
    function updatePrivacyConfig(
        uint256 _minDataSize,
        uint256 _noiseMagnitude,
        bool _requireEncryption,
        uint256 _minReputation
    ) external onlyCoordinator {
        privacyConfig.minDataSize = _minDataSize;
        privacyConfig.noiseMagnitude = _noiseMagnitude;
        privacyConfig.requireEncryption = _requireEncryption;
        privacyConfig.minReputation = _minReputation;
        
        emit PrivacyConfigUpdated(_minDataSize, _noiseMagnitude);
    }
    
    // ==================== View Functions ====================
    
    /**
     * @dev Get model information for a specific round
     */
    function getModelInfo(uint256 _round) external view returns (
        string memory ipfsHash,
        bytes32 modelHash,
        uint256 timestamp,
        uint256 numContributors,
        bool isAggregated
    ) {
        ModelVersion storage model = globalModels[_round];
        return (
            model.ipfsHash,
            model.modelHash,
            model.timestamp,
            model.numContributors,
            model.isAggregated
        );
    }
    
    /**
     * @dev Get contributors for a specific round
     */
    function getRoundContributors(uint256 _round) external view returns (address[] memory) {
        return globalModels[_round].contributors;
    }
    
    /**
     * @dev Get client information
     */
    function getClientInfo(address _client) external view returns (
        bool isActive,
        uint256 reputation,
        uint256 totalContributions,
        uint256 stakedAmount
    ) {
        Client storage client = clients[_client];
        return (
            client.isActive,
            client.reputation,
            client.totalContributions,
            client.stakedAmount
        );
    }
    
    /**
     * @dev Get local update information
     */
    function getLocalUpdate(uint256 _round, address _client) external view returns (
        string memory ipfsHash,
        bytes32 updateHash,
        uint256 timestamp,
        uint256 dataSize,
        bool isVerified
    ) {
        LocalUpdate storage update = localUpdates[_round][_client];
        return (
            update.ipfsHash,
            update.updateHash,
            update.timestamp,
            update.dataSize,
            update.isVerified
        );
    }
    
    /**
     * @dev Get active clients count
     */
    function getActiveClientsCount() external view returns (uint256) {
        uint256 count = 0;
        for (uint256 i = 0; i < registeredClients.length; i++) {
            if (clients[registeredClients[i]].isActive) {
                count++;
            }
        }
        return count;
    }
    
    // ==================== Emergency Functions ====================
    
    /**
     * @dev Emergency pause (only coordinator)
     */
    function emergencyPause() external onlyCoordinator {
        roundTimeout = 0;
    }
    
    /**
     * @dev Transfer coordinator role
     */
    function transferCoordinator(address _newCoordinator) external onlyCoordinator {
        require(_newCoordinator != address(0), "Invalid address");
        coordinator = _newCoordinator;
    }
}
