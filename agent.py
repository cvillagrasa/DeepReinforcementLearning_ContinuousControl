from dataclasses import dataclass
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from model import NetworkWithTarget, Actor, Critic


@dataclass
class CriticHelper:
    """Object to abstract the Critic setup, in order to be able to inherit both from single and multi-agent objects"""
    state_size: int
    action_size: int
    n_hidden_critic: int = None
    lr_critic: float = 1e-4  # learning rate for the critic
    weight_decay_critic: float = 0  # weight decay for the critic
    tau: float = 1e-3  # for soft update of target parameters
    optim_func_critic: ... = None

    def __post_init__(self):
        if self.optim_func_critic is None:
            self.optim_func_critic = optim.Adam
            self.optim_params_critic = {'lr': self.lr_critic, 'weight_decay': self.weight_decay_critic}

    @property
    def common_params(self):
        return {'state_size': self.state_size, 'action_size': self.action_size}

    @property
    def critic_params(self):
        return {**self.common_params, 'n_hidden': self.n_hidden_critic}

    def setup_critic(self):
        self.critic = NetworkWithTarget(
            Critic, params=self.critic_params, optim_func=self.optim_func_critic, optim_params=self.optim_params_critic,
            tau=self.tau, device=self.device
        )
        self.critics = [self.critic]
        common_data = {'critic': self.critic}
        if self.twin_delayed:
            self.critic2 = NetworkWithTarget(
                Critic, params=self.critic_params, optim_func=self.optim_func_critic,
                optim_params=self.optim_params_critic, tau=self.tau, device=self.device
            )
            self.critics.append(self.critic2)
            common_data['critic2'] = self.critic2
        common_data['critics'] = self.critics
        return common_data


@dataclass
class MultiAgent(CriticHelper):
    agent_class: ... = None
    num_agents: int = 1
    buffer_size: int = int(1e5)  # replay buffer size
    batch_size: int = 128  # minibatch size
    prioritized_experience_replay: ... = 0.  # 0 -> pure random sampling, 1 -> full prioritized experience replay
    twin_delayed: bool = False  # whether to implement the TD3 algorithm, instead of vanilla DDPG
    beta: float = 0.5  # non-uniform probabilities compensation in importance sampling for prioritized experience replay
    device: ... = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __post_init__(self):
        super().__post_init__()
        self.memory_replay = MemoryReplayBuffer(
            self.action_size, self.buffer_size, self.batch_size, a=self.prioritized_experience_replay,
            beta=self.beta, device=self.device
        )
        critic_common_data = self.setup_critic()
        agent_params = {'state_size': self.state_size, 'action_size': self.action_size,
                        'buffer_size': self.buffer_size, 'batch_size': self.batch_size,
                        'prioritized_experience_replay': self.prioritized_experience_replay,
                        'twin_delayed': self.twin_delayed, 'beta': self.beta, 'device': self.device,
                        'memory_replay': self.memory_replay, 'critic_common_data': critic_common_data}
        self.agents = [self.agent_class(**agent_params) for _ in range(self.num_agents)]

    def __call__(self):
        return self.agents


@dataclass
class AgentDDPG(CriticHelper):
    buffer_size: int = int(1e5)  # replay buffer size
    batch_size: int = 256  # minibatch size
    gamma: float = 0.99  # discount factor
    lr_actor: float = 3e-4  # learning rate for the actor  3e-4
    update_every: ... = 1  # how often to update the network
    prioritized_experience_replay: ... = 0.  # 0 -> pure random sampling, 1 -> full prioritized experience replay
    twin_delayed: bool = False  # whether to implement the TD3 algorithm, instead of vanilla DDPG
    delayed_policy_factor: int = 2  # amount of times the Q network is more often updated than the policy
    e: float = 1e-4  # constant to prevent samples from being starved in prioritized experience replay
    beta: float = 0.5  # non-uniform probabilities compensation in importance sampling for prioritized experience replay
    device: ... = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    warmup: float = 1000  # warmup timesteps taking random actions
    noise_exploration: float = 0.3  # noise on selected actions
    noise_policy: float = 0.2  # noise on target actions (only applicable if TD3)
    noise_policy_clipping: float = 0.3  # clipping parameter for the policy noise
    optim_func_actor: ... = None
    n_hidden_actor: tuple = None
    critic_common_data: ... = None
    memory_replay: ... = None

    def __post_init__(self):
        super().__post_init__()
        if self.optim_func_actor is None:
            self.optim_func_actor = optim.Adam
        self.optim_params_actor = {'lr': self.lr_actor}
        self.actor = NetworkWithTarget(
            Actor, params=self.actor_params, optim_func=self.optim_func_actor, optim_params=self.optim_params_actor,
            tau=self.tau, device=self.device
        )
        if self.critic_common_data is None:
            _ = self.setup_critic()
        else:
            for name, obj in self.critic_common_data.items():
                setattr(self, name, obj)
        if self.memory_replay is None:
            self.memory_replay = MemoryReplayBuffer(
                self.action_size, self.buffer_size, self.batch_size, a=self.prioritized_experience_replay,
                beta=self.beta, device=self.device
            )
        self.t_step = 0

    @property
    def common_params(self):
        return {'state_size': self.state_size, 'action_size': self.action_size}

    @property
    def actor_params(self):
        return {**self.common_params, 'n_hidden': self.n_hidden_actor}

    @property
    def critic_params(self):
        return {**self.common_params, 'n_hidden': self.n_hidden_critic}

    def step(self, state, action, reward, next_state, done):
        self.t_step += 1
        if self.memory_replay.prioritized:
            experience = self.memory_replay.experience(state, action, reward, next_state, done, 1)
            experience = self.memory_replay.experiences_to_tensors([experience])
            state, action, reward, next_state, done, priority = experience
            pred_qvalues_all_critics = [critic.local(state, action) for critic in self.critics]
            computed_qvalues = self.computed_qvalues_critic(reward, next_state, done)
            td_errors = [self.temporal_difference_error(pred_qvalues, computed_qvalues).abs().item()
                         for pred_qvalues in pred_qvalues_all_critics]
            td_error = np.min(td_errors)
            priority = td_error + self.e
        else:
            priority = 1.
        self.memory_replay.add(state, action, reward, next_state, done, priority)
        if self.t_step % self.update_every == 0 and len(self.memory_replay) > self.batch_size:
            sample = self.memory_replay.sample()
            self.learn(sample['experiences'], sample['max_weight'])

    def action(self, state, noise=True):
        if self.t_step < self.warmup:
            return np.random.rand(self.action_size) * 2 - 1
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.actor.local.eval()
            with torch.no_grad():
                action_probs = self.actor.local(state).cpu().data.numpy()
            self.actor.local.train()
            if noise:
                action_probs += np.random.normal(size=self.action_size, scale=self.noise_exploration)
            return np.clip(action_probs, -1, 1)

    def learn(self, experiences, max_weight):
        states, actions, rewards, next_states, dones, priorities = experiences
        self.learn_critic(states, actions, rewards, next_states, dones, priorities, max_weight)
        if not self.twin_delayed or self.t_step % self.delayed_policy_factor == 0:
            self.learn_policy(states)

    def learn_policy(self, states):
        self.learn_actor(states)
        for critic in self.critics:
            critic.soft_update()
        self.actor.soft_update()

    def computed_qvalues_critic(self, rewards, next_states, dones):
        actions_next = self.actor.target(next_states)
        if self.twin_delayed:
            noise = np.random.normal(scale=self.noise_policy)
            noise = np.clip(noise, -self.noise_policy_clipping, self.noise_policy_clipping)
            actions_next = actions_next + torch.from_numpy(np.array(noise)).float().to(self.device)
        actions_next = torch.clamp(actions_next, -1, 1)
        qvalues_next = [critic.target(next_states, actions_next) for critic in self.critics]
        qvalues_next = torch.stack(qvalues_next, dim=1).squeeze(-1)
        qvalues_next = torch.min(qvalues_next, dim=1)[0]  # minimum of twin critics, if applicable
        return rewards + self.gamma * qvalues_next.unsqueeze(-1) * (1 - dones)

    def learn_critic(self, states, actions, rewards, next_states, dones, priorities, max_weight):
        pred_qvalues_all_critics = [critic.local(states, actions) for critic in self.critics]
        computed_qvalues = self.computed_qvalues_critic(rewards, next_states, dones)
        if self.memory_replay.prioritized:
            weight = ((len(self.memory_replay) * priorities) ** -self.beta / max_weight).squeeze()
            pred_qvalues_all_critics = [value * weight for value in pred_qvalues_all_critics]
            computed_qvalues *= weight.unsqueeze(-1)
        critic_loss = []
        for critic, pred_qvalues in zip(self.critics, pred_qvalues_all_critics):
            critic.loss = F.mse_loss(pred_qvalues, computed_qvalues)
            critic_loss.append(critic.loss)
            critic.optimizer.zero_grad()
        critic_loss = np.sum(critic_loss)
        critic_loss.backward()
        for critic in self.critics:
            critic.optimizer.step()

    def learn_actor(self, states):
        actions_pred = self.actor.local(states)
        self.actor.loss = -self.critic.local(states, actions_pred).mean()
        self.actor.optimizer.zero_grad()
        self.actor.loss.backward()
        self.actor.optimizer.step()

    def temporal_difference_error(self, pred_qvalues, computed_qvalues):
        return computed_qvalues - pred_qvalues

    @property
    def critic_loss(self):
        return np.mean([critic.loss.cpu().data.numpy() for critic in self.critics])


@dataclass
class MemoryReplayBuffer:
    action_size: ...
    buffer_size: ...
    batch_size: ...
    device: ... = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    a: ... = 0.  # 0 -> pure random sampling, 1 -> full prioritized experience replay
    beta: ... = 0.5  # non-uniform probabilities compensation in importance sampling for prioritized experience replay

    def __post_init__(self):
        self.buffer = deque(maxlen=self.buffer_size)
        self.field_names = ['state', 'action', 'reward', 'next_state', 'done', 'priority']
        self.experience = namedtuple('Experience', field_names=self.field_names)

    def add(self, state, action, reward, next_state, done, priority=1.):
        e = self.experience(state, action, reward, next_state, done, priority)
        self.buffer.append(e)

    def sample(self):
        if self.prioritized:
            denominator = self.sum_priorities
            probabilities = [e.priority ** self.a / denominator for e in self.buffer]
            sample_indexes = np.random.choice(np.arange(len(self.buffer)), size=self.batch_size, p=probabilities)
            experiences = [self.buffer[idx] for idx in sample_indexes]
            max_weight = (np.min(probabilities) * len(self)) ** -self.beta
        else:
            experiences = random.sample(self.buffer, self.batch_size)
            max_weight = 1.
        return {'experiences': self.experiences_to_tensors(experiences), 'max_weight': max_weight}

    def experiences_to_tensors(self, experiences):
        experiences = [[e[idx] for e in experiences if e is not None] for idx, field in enumerate(self.field_names)]
        experiences = [np.vstack(e) for e in experiences]
        experiences[4] = experiences[4].astype(np.uint8)
        experiences = [torch.from_numpy(e) for e in experiences]
        experiences = [e.float() for idx, e in enumerate(experiences)]
        return [e.to(self.device) for e in experiences]

    def __len__(self):
        return len(self.buffer)

    @property
    def prioritized(self):
        return self.a > 0

    @property
    def sum_priorities(self):
        return np.sum([e.priority ** self.a for e in self.buffer])
