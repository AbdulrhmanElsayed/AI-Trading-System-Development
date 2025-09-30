"""
Reinforcement Learning Agent

RL-based trading agent for strategy optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime
from collections import deque

from src.utils.config import ConfigManager
from src.utils.logger import TradingLogger


class RLAgent:
    """Reinforcement Learning agent for trading."""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = TradingLogger("RLAgent")
        
        # RL Configuration
        self.algorithm = config.get('ml.rl.algorithm', 'PPO')
        self.learning_rate = config.get('ml.rl.learning_rate', 0.001)
        self.reward_function = config.get('ml.rl.reward_function', 'sharpe_ratio')
        
        # Environment setup
        self.state_size = config.get('ml.rl.state_size', 50)
        self.action_size = 3  # Buy, Hold, Sell
        
        # Experience replay
        self.memory_size = config.get('ml.rl.memory_size', 10000)
        self.experience_buffer = deque(maxlen=self.memory_size)
        
        # Training parameters
        self.batch_size = config.get('ml.rl.batch_size', 32)
        self.gamma = config.get('ml.rl.gamma', 0.95)  # Discount factor
        self.epsilon = config.get('ml.rl.epsilon', 1.0)  # Exploration rate
        self.epsilon_decay = config.get('ml.rl.epsilon_decay', 0.995)
        self.epsilon_min = config.get('ml.rl.epsilon_min', 0.01)
        
        # Model
        self.agent = None
        self.is_initialized = False
        
        # State tracking
        self.current_state = None
        self.last_action = None
        self.last_reward = 0.0
        
        # Performance tracking
        self.episode_rewards = []
        self.total_episodes = 0
    
    async def initialize(self):
        """Initialize RL agent."""
        try:
            self.logger.info(f"Initializing RL agent with {self.algorithm} algorithm...")
            
            if self.algorithm == 'PPO':
                await self._initialize_ppo_agent()
            elif self.algorithm == 'DQN':
                await self._initialize_dqn_agent()
            elif self.algorithm == 'A3C':
                await self._initialize_a3c_agent()
            else:
                raise ValueError(f"Unsupported RL algorithm: {self.algorithm}")
            
            self.is_initialized = True
            self.logger.info("RL agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RL agent: {e}")
            raise
    
    async def _initialize_ppo_agent(self):
        """Initialize PPO agent (placeholder)."""
        # This would use a proper RL library like stable-baselines3
        # For now, implement a simple Q-learning approximation
        self.q_table = np.random.rand(self.state_size, self.action_size) * 0.1
        self.logger.info("PPO agent initialized (simplified implementation)")
    
    async def _initialize_dqn_agent(self):
        """Initialize DQN agent (placeholder)."""
        self.q_table = np.random.rand(self.state_size, self.action_size) * 0.1
        self.logger.info("DQN agent initialized (simplified implementation)")
    
    async def _initialize_a3c_agent(self):
        """Initialize A3C agent (placeholder)."""
        self.q_table = np.random.rand(self.state_size, self.action_size) * 0.1
        self.logger.info("A3C agent initialized (simplified implementation)")
    
    async def get_action(self, state: np.ndarray) -> int:
        """Get action from RL agent."""
        try:
            if not self.is_initialized:
                return 1  # Default to HOLD
            
            # Convert state to discrete state index (simplified)
            state_index = self._discretize_state(state)
            
            # Epsilon-greedy action selection
            if np.random.random() <= self.epsilon:
                # Exploration: random action
                action = np.random.randint(0, self.action_size)
            else:
                # Exploitation: best action from Q-table
                action = np.argmax(self.q_table[state_index])
            
            self.last_action = action
            self.current_state = state_index
            
            return action
            
        except Exception as e:
            self.logger.error(f"Error getting RL action: {e}")
            return 1  # Default to HOLD
    
    def _discretize_state(self, state: np.ndarray) -> int:
        """Convert continuous state to discrete state index."""
        # Simple discretization - in practice, you'd use more sophisticated methods
        if len(state) == 0:
            return 0
        
        # Use hash of state values to create index
        state_hash = hash(tuple(np.round(state, 2)))
        return abs(state_hash) % self.state_size
    
    async def learn(self, reward: float, next_state: np.ndarray, done: bool = False):
        """Update agent based on experience."""
        try:
            if not self.is_initialized or self.current_state is None:
                return
            
            # Store experience
            experience = {
                'state': self.current_state,
                'action': self.last_action,
                'reward': reward,
                'next_state': self._discretize_state(next_state),
                'done': done
            }
            
            self.experience_buffer.append(experience)
            
            # Q-learning update (simplified)
            if self.last_action is not None:
                next_state_index = self._discretize_state(next_state)
                
                # Q-learning formula
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * np.max(self.q_table[next_state_index])
                
                # Update Q-value
                current_q = self.q_table[self.current_state][self.last_action]
                new_q = current_q + self.learning_rate * (target - current_q)
                self.q_table[self.current_state][self.last_action] = new_q
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            self.last_reward = reward
            
            # Periodic batch learning
            if len(self.experience_buffer) >= self.batch_size and len(self.experience_buffer) % 100 == 0:
                await self._batch_learn()
            
        except Exception as e:
            self.logger.error(f"Error in RL learning: {e}")
    
    async def _batch_learn(self):
        """Learn from batch of experiences."""
        if len(self.experience_buffer) < self.batch_size:
            return
        
        # Sample random batch
        batch_indices = np.random.choice(
            len(self.experience_buffer), 
            self.batch_size, 
            replace=False
        )
        
        batch_experiences = [self.experience_buffer[i] for i in batch_indices]
        
        # Batch Q-learning updates
        for exp in batch_experiences:
            state = exp['state']
            action = exp['action']
            reward = exp['reward']
            next_state = exp['next_state']
            done = exp['done']
            
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.q_table[next_state])
            
            current_q = self.q_table[state][action]
            new_q = current_q + self.learning_rate * (target - current_q)
            self.q_table[state][action] = new_q
    
    def calculate_reward(
        self, 
        action: int, 
        price_change: float, 
        portfolio_value: float,
        benchmark_return: float = 0.0
    ) -> float:
        """Calculate reward based on action and market movement."""
        try:
            if self.reward_function == 'simple_return':
                return self._simple_return_reward(action, price_change)
            elif self.reward_function == 'sharpe_ratio':
                return self._sharpe_ratio_reward(action, price_change, portfolio_value)
            elif self.reward_function == 'risk_adjusted':
                return self._risk_adjusted_reward(action, price_change, portfolio_value)
            else:
                return self._simple_return_reward(action, price_change)
                
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def _simple_return_reward(self, action: int, price_change: float) -> float:
        """Simple return-based reward."""
        if action == 0:  # Sell
            return -price_change  # Profit when price goes down
        elif action == 2:  # Buy
            return price_change   # Profit when price goes up
        else:  # Hold
            return 0.0
    
    def _sharpe_ratio_reward(self, action: int, price_change: float, portfolio_value: float) -> float:
        """Sharpe ratio-based reward."""
        # Simplified Sharpe ratio calculation
        base_reward = self._simple_return_reward(action, price_change)
        
        # Add risk adjustment
        if len(self.episode_rewards) > 10:
            returns = np.array(self.episode_rewards[-10:])
            volatility = np.std(returns) if np.std(returns) > 0 else 1.0
            risk_adjusted_reward = base_reward / volatility
        else:
            risk_adjusted_reward = base_reward
        
        return risk_adjusted_reward
    
    def _risk_adjusted_reward(self, action: int, price_change: float, portfolio_value: float) -> float:
        """Risk-adjusted reward function."""
        base_reward = self._simple_return_reward(action, price_change)
        
        # Penalize large positions or high volatility
        volatility_penalty = 0.0
        if len(self.episode_rewards) > 5:
            recent_volatility = np.std(self.episode_rewards[-5:])
            volatility_penalty = -0.1 * recent_volatility
        
        return base_reward + volatility_penalty
    
    async def start_episode(self):
        """Start a new trading episode."""
        self.total_episodes += 1
        self.current_state = None
        self.last_action = None
        self.last_reward = 0.0
        
        self.logger.info(f"Starting RL episode {self.total_episodes}")
    
    async def end_episode(self, final_reward: float):
        """End current episode and update statistics."""
        self.episode_rewards.append(final_reward)
        
        # Keep only recent episodes for memory efficiency
        if len(self.episode_rewards) > 1000:
            self.episode_rewards = self.episode_rewards[-1000:]
        
        avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
        
        self.logger.info(
            f"Episode {self.total_episodes} completed. "
            f"Reward: {final_reward:.4f}, "
            f"Avg reward (last 100): {avg_reward:.4f}, "
            f"Epsilon: {self.epsilon:.4f}"
        )
    
    async def stop(self):
        """Stop RL agent."""
        self.logger.info("Stopping RL agent...")
        
        # Save final statistics
        if self.episode_rewards:
            self.logger.info(
                f"RL Agent final stats - Episodes: {self.total_episodes}, "
                f"Average reward: {np.mean(self.episode_rewards):.4f}"
            )
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get RL agent statistics."""
        stats = {
            'algorithm': self.algorithm,
            'total_episodes': self.total_episodes,
            'epsilon': self.epsilon,
            'experience_buffer_size': len(self.experience_buffer),
            'is_initialized': self.is_initialized
        }
        
        if self.episode_rewards:
            stats.update({
                'total_rewards': len(self.episode_rewards),
                'average_reward': np.mean(self.episode_rewards),
                'best_reward': np.max(self.episode_rewards),
                'worst_reward': np.min(self.episode_rewards),
                'recent_average_reward': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
            })
        
        return stats