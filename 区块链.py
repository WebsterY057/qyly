import hashlib
import time
from datetime import datetime


class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index  # 区块在链中的位置
        self.previous_hash = previous_hash  # 前一个区块的哈希值
        self.timestamp = timestamp  # 区块创建时间戳
        self.data = data  # 区块存储的数据（例如交易信息）
        self.hash = hash  # 当前区块的哈希值


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]  # 初始化区块链，包含创世区块

    def create_genesis_block(self):
        # 创建区块链的第一个区块（创世区块）
        return Block(0, "0", time.time(), "Genesis Block", self.calculate_hash(0, "0", time.time(), "Genesis Block"))

    def calculate_hash(self, index, previous_hash, timestamp, data):
        # 计算区块的SHA-256哈希值
        value = str(index) + str(previous_hash) + str(timestamp) + str(data)
        return hashlib.sha256(value.encode('utf-8')).hexdigest()

    def add_block(self, data):
        # 向区块链添加新区块
        previous_block = self.chain[-1]
        index = previous_block.index + 1
        timestamp = time.time()
        hash = self.calculate_hash(index, previous_block.hash, timestamp, data)

        new_block = Block(index, previous_block.hash, timestamp, data, hash)
        self.chain.append(new_block)

        print(f"区块 #{index} 已添加到区块链")
        print(f"哈希: {hash}")

    def is_chain_valid(self):
        # 验证区块链的完整性
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # 检查当前区块的哈希是否正确
            if current_block.hash != self.calculate_hash(
                    current_block.index, current_block.previous_hash,
                    current_block.timestamp, current_block.data
            ):
                return False

            # 检查当前区块是否指向正确的前一个区块
            if current_block.previous_hash != previous_block.hash:
                return False

        return True


# 使用示例
my_blockchain = Blockchain()

# 添加一些区块
my_blockchain.add_block("交易1: Alice向Bob转账10BTC")
my_blockchain.add_block("交易2: Charlie向David转账5BTC")

# 验证区块链
print("区块链是否有效:", my_blockchain.is_chain_valid())

# 尝试篡改数据（演示不可篡改性）
my_blockchain.chain[1].data = "篡改的交易数据"
print("篡改后区块链是否有效:", my_blockchain.is_chain_valid())

#####智能合约进阶 - 去中心化投票系统
"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DecentralizedVoting {
    struct Candidate {
        uint id;
        string name;
        uint voteCount;
    }
    
    struct Voter {
        bool voted;  // 是否已投票
        uint vote;   // 投票的候选人ID
        uint weight; // 投票权重（可用于代币加权投票）
    }
    
    address public owner;  // 合约所有者
    string public votingName;  // 投票名称
    mapping(uint => Candidate) public candidates;  // 候选人映射
    mapping(address => Voter) public voters;  // 投票人映射
    uint public candidatesCount;  // 候选人数量
    uint public votingEndTime;  // 投票结束时间
    bool public votingActive;  // 投票是否活跃
    
    event VoteCast(address indexed voter, uint candidateId);  // 投票事件
    event VotingStarted(string votingName, uint endTime);  // 投票开始事件
    event VotingEnded(uint winningCandidateId);  // 投票结束事件
    
    modifier onlyOwner() {
        require(msg.sender == owner, "只有所有者可以执行此操作");
        _;
    }
    
    modifier duringVoting() {
        require(votingActive, "投票未开始");
        require(block.timestamp <= votingEndTime, "投票已结束");
        _;
    }
    
    constructor(string memory _votingName) {
        owner = msg.sender;
        votingName = _votingName;
    }
    
    // 添加候选人
    function addCandidate(string memory _name) public onlyOwner {
        require(!votingActive, "投票已开始，不能添加候选人");
        candidatesCount++;
        candidates[candidatesCount] = Candidate(candidatesCount, _name, 0);
    }
    
    // 开始投票
    function startVoting(uint _durationInMinutes) public onlyOwner {
        require(candidatesCount > 0, "至少需要一名候选人");
        require(!votingActive, "投票已在进行中");
        
        votingActive = true;
        votingEndTime = block.timestamp + (_durationInMinutes * 1 minutes);
        
        emit VotingStarted(votingName, votingEndTime);
    }
    
    // 投票
    function vote(uint _candidateId) public duringVoting {
        Voter storage sender = voters[msg.sender];
        require(!sender.voted, "你已经投过票了");
        require(_candidateId > 0 && _candidateId <= candidatesCount, "无效的候选人ID");
        
        sender.voted = true;
        sender.vote = _candidateId;
        sender.weight = 1; // 默认权重为1
        
        candidates[_candidateId].voteCount += sender.weight;
        
        emit VoteCast(msg.sender, _candidateId);
    }
    
    // 结束投票并宣布获胜者
    function endVoting() public onlyOwner returns (uint) {
        require(votingActive, "投票未开始");
        require(block.timestamp > votingEndTime, "投票尚未结束");
        
        votingActive = false;
        
        uint winningCandidateId = getWinningCandidate();
        emit VotingEnded(winningCandidateId);
        
        return winningCandidateId;
    }
    
    // 获取获胜候选人
    function getWinningCandidate() public view returns (uint) {
        uint winningVoteCount = 0;
        uint winningCandidateId = 0;
        
        for (uint i = 1; i <= candidatesCount; i++) {
            if (candidates[i].voteCount > winningVoteCount) {
                winningVoteCount = candidates[i].voteCount;
                winningCandidateId = i;
            }
        }
        
        return winningCandidateId;
    }
    
    // 获取投票结果
    function getResults() public view returns (uint[] memory, uint[] memory) {
        uint[] memory candidateIds = new uint[](candidatesCount);
        uint[] memory voteCounts = new uint[](candidatesCount);
        
        for (uint i = 0; i < candidatesCount; i++) {
            candidateIds[i] = candidates[i+1].id;
            voteCounts[i] = candidates[i+1].voteCount;
        }
        
        return (candidateIds, voteCounts);
    }
}"""