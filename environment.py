from dataclasses import dataclass
from pathlib import Path
from collections import deque
from itertools import count
import numpy as np
import pandas as pd
import torch
from unityagents import UnityEnvironment
from agent import AgentDDPG, MultiAgent


@dataclass
class ReacherEnvironment:
    env_path_single_agent: ... = Path('./ReacherLinuxSingleAgent/Reacher.x86_64')
    env_path_multi_agent: ... = Path('./ReacherLinuxMultiAgent/Reacher.x86_64')
    env = None
    action_space_size = None
    observation_space_size = None
    multiagent: bool = False  # either to use the multi-agent environment, or the single-agent instead
    agent_params: dict = None  # extra parameters for the agent are passed as a dictionary
    base_port: int = 5005
    train_mode: bool = True  # True for training / False for inference // removed in current version of ML-Agents
    preferred_device: str = None  # pass "cpu" to force running on the cpu

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def load(self):
        env_path = self.env_path_multi_agent if self.multiagent else self.env_path_single_agent
        self.env = UnityEnvironment(file_name=str(env_path), base_port=self.base_port)
        self.num_agents, self.observation_space_size = self.get_observation_space_size()
        self.action_space_size = self.get_action_space_size()
        if self.preferred_device is not None:
            self.device = torch.device(self.preferred_device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.agent_params = {} if self.agent_params is None else self.agent_params
        self.agents = MultiAgent(agent_class=AgentDDPG, num_agents=self.num_agents,
                                 state_size=self.observation_space_size, action_size=self.action_space_size,
                                 device=self.device, **self.agent_params)

    def close(self):
        self.env.close()
        self.env = None
        self.agents = None

    def reset(self):
        return self.env.reset(train_mode=self.train_mode)

    @property
    def default_brain_name(self):
        return self.env.brain_names[0]

    @property
    def default_brain(self):
        return self.env.brains[self.default_brain_name]

    def get_action_space_size(self):
        return self.default_brain.vector_action_space_size

    def get_observation_space_size(self):
        env_info = self.reset()
        vector_observations = env_info[self.default_brain_name].vector_observations
        num_agents, observation_space_size = vector_observations.shape
        return num_agents, observation_space_size

    def select_random_action(self):
        return np.random.randint(self.action_space_size)

    def train(self, n_episodes=2000, max_t=np.inf, metrics_log_size=100, save=False, saveas='checkpoint'):
        scores = pd.DataFrame()
        max_score = 0
        scores_window = deque(maxlen=metrics_log_size)
        actor_loss_window = deque(maxlen=metrics_log_size)
        critic_loss_window = deque(maxlen=metrics_log_size)
        for episode in range(n_episodes):
            score = self.train_episode(max_t)
            mean_score = np.mean(score)
            scores_window.append(mean_score)
            scores = pd.concat([scores, pd.DataFrame(score).T], ignore_index=True)
            if self.train_mode:
                actor_loss_window.append(np.mean([float(agent.actor.loss) for agent in self.agents()]))
                critic_loss_window.append(np.mean([float(agent.critic_loss) for agent in self.agents()]))
            self.print_episode_info(episode, scores_window, actor_loss_window, critic_loss_window, metrics_log_size)
            if self.train_mode and save and mean_score > max_score:
                self.save_parameters(saveas)
            max_score = mean_score if mean_score > max_score else max_score
        return scores

    def train_episode(self, max_t):
        env_info = self.reset()[self.default_brain_name]
        states = env_info.vector_observations
        scores = np.zeros(self.num_agents)
        timesteps = count() if max_t is np.inf else range(max_t)
        for t in timesteps:
            rewards, next_states, dones = self.train_step(states)
            scores += np.array(rewards)
            states = next_states
            if any(dones):
                break
        return scores

    def train_step(self, states):
        actions = [agent.action(state) for agent, state in zip(self.agents(), states)]
        env_info = self.env.step(np.array(actions).flatten())[self.default_brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        if self.train_mode:
            for agent, state, action, reward, next_state, done in zip(
                    self.agents(), states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)
        return rewards, next_states, dones

    def play(self, n_episodes=2000, max_t=10000, metrics_log_size=1):
        self.train_mode = False
        return self.train(n_episodes=n_episodes, max_t=max_t, metrics_log_size=metrics_log_size)

    def print_episode_info(self, episode, scores_window, actor_loss_window, critic_loss_window, metrics_log_size):
        line = self.episode_line(episode, scores_window, actor_loss_window, critic_loss_window)
        print(line, end='')
        if episode % metrics_log_size == 0:
            print(line)

    def episode_line(self, episode, scores_window, actor_loss_window, critic_loss_window):
        episode += 1
        line = f'\rEpisode {episode}'
        line += f'\tAverage Score: {np.mean(scores_window):.2f}'
        if self.train_mode:
            line += f'\tActor loss: {np.mean(actor_loss_window)}'
            line += f'\tCritic loss: {np.mean(critic_loss_window)}'
        return line

    def save_parameters(self, saveas='checkpoint'):
        savepath = Path() / 'checkpoints'
        if self.multiagent:
            savepath = savepath / saveas
        savepath.mkdir(exist_ok=True, parents=True)
        for idx, agent in enumerate(self.agents()):
            suffix = f'_{idx:2d}' if self.multiagent else ''
            torch.save(agent.actor.local.state_dict(), savepath / f'{saveas}{suffix}.pt')

    def load_parameters(self, filename_base):
        filename_base = filename_base / filename_base.name if self.multiagent else filename_base
        suffixes = [f'_{idx:2d}' for idx in range(self.num_agents)] if self.multiagent else ['']
        for agent, suffix in zip(self.agents(), suffixes):
            unpickled_state_dict = torch.load(f'{filename_base}{suffix}.pt')
            agent.actor.local.load_state_dict(unpickled_state_dict)
            agent.actor.local.eval()
        self.train_mode = False
