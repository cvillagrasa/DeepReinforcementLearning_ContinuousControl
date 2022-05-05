from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


@dataclass
class NetworkWithTarget:
    network: ...
    params: ... = None
    optim_func: ... = optim.Adam
    optim_params: ... = None
    tau: float = 1e-3
    device: ... = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    loss: ... = None

    def __post_init__(self):
        if self.optim_params is None:
            self.optim_params = {'lr': 1e-4}
        self.local = self.network(**self.params).to(self.device)
        self.target = self.network(**self.params).to(self.device)
        self.optimizer = self.optim_func(self.local.parameters(), **self.optim_params)
        self.soft_update(tau=1.)

    def soft_update(self, tau=None):
        tau = self.tau if tau is None else tau
        for local_param, target_param in zip(self.local.parameters(), self.target.parameters()):
            target_param.data.copy_(tau * local_param.data + (1. - tau) * target_param.data)


@dataclass
class NetworkPattern(nn.Module, ABC):
    state_size: int
    action_size: int
    n_hidden: tuple = None

    def __post_init__(self):
        super().__init__()
        self.n_hidden = (400, 300) if self.n_hidden is None else self.n_hidden
        if not isinstance(self.n_hidden, tuple) and not isinstance(self.n_hidden, list) or len(self.n_hidden) != 2:
            raise ValueError(f'The "n_hidden parameter" must be a tuple/list of two elements')
        self.linear1 = nn.Linear(self.state_size, self.n_hidden[0])
        self.linear2 = self.define_second_layer()
        self.linear3 = self.define_third_layer()
        self.reset_parameters()

    @abstractmethod
    def define_second_layer(self):
        ...

    @abstractmethod
    def define_third_layer(self):
        ...

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.uniform_(self.linear3.weight, *self.last_layer_init_limits)

    def forward_base(self, state):
        @try_various_attempts(2)
        def first_layer(state):
            return F.relu(self.linear1(state))
        return first_layer(state)

    @abstractmethod
    def forward(self, *args):
        ...  # Second layer onwards work well if the first one already has returned output


@dataclass
class Actor(NetworkPattern):
    last_layer_init_limits: tuple = (-3e-3, 3e-3)

    def define_second_layer(self):
        return nn.Linear(self.n_hidden[0], self.n_hidden[1])

    def define_third_layer(self):
        return nn.Linear(self.n_hidden[1], self.action_size)

    def forward(self, state):
        x = super().forward_base(state)
        x = F.relu(self.linear2(x))
        return F.tanh(self.linear3(x))


@dataclass
class Critic(NetworkPattern):
    last_layer_init_limits: tuple = (-3e-4, 3e-4)

    def define_second_layer(self):
        return nn.Linear(self.n_hidden[0] + self.action_size, self.n_hidden[1])

    def define_third_layer(self):
        return nn.Linear(self.n_hidden[1], 1)

    def forward(self, state, action):
        x = super().forward_base(state)
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.linear2(x))
        return self.linear3(x)


def try_various_attempts(allowed_attempts):
    """Dirty hack to prevent PyTorch 0.4 to halt at start on a RTX2070 Super with Cuda 9.0"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(allowed_attempts):
                try:
                    result = func(*args, **kwargs)
                    break
                except RuntimeError:  # When x.is_cuda, it usually just works on second attempt
                    if attempt + 1 == allowed_attempts:
                        raise RuntimeError(
                            f'RuntimeError: {allowed_attempts} successive failed attempts : ' +
                            f'Probably cublas runtime error'
                        )
                    else:
                        pass
            return result
        return wrapper
    return decorator
