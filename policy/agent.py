import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actor_out = self.actor(state)
        value = self.critic(state)
        return actor_out, value


@dataclass
class ActionOutput:
    proportions: torch.Tensor  # shape: (action_size,)
    durations: np.ndarray  # shape: (action_size,)
    log_prob: Optional[torch.Tensor]
    value: Optional[torch.Tensor]
    entropy: Optional[torch.Tensor]


class PolicyGradientAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    @staticmethod
    def _proportions_to_durations(
        proportions: torch.Tensor,
        *,
        total_time: int,
        min_green: int,
    ) -> np.ndarray:
        if total_time <= 0:
            raise ValueError("total_time must be > 0")
        if min_green < 0:
            raise ValueError("min_green must be >= 0")

        n = int(proportions.shape[0])
        if n <= 0:
            raise ValueError("action_size must be > 0")

        base = int(min_green) * n
        if base > total_time:
            raise ValueError(
                f"min_green too large: {min_green}s * {n} phases > total_time {total_time}s"
            )

        remainder = total_time - base

        p = proportions.detach().cpu().numpy().astype(np.float64)
        p = np.clip(p, 1e-8, None)
        p = p / p.sum()

        raw = p * remainder
        floored = np.floor(raw).astype(int)
        leftover = remainder - int(floored.sum())

        if leftover > 0:
            frac = raw - floored
            order = np.argsort(-frac)
            for i in range(leftover):
                floored[order[i % n]] += 1

        durations = floored + min_green
        # Safety: ensure exact sum.
        diff = total_time - int(durations.sum())
        if diff != 0:
            durations[np.argmax(durations)] += diff

        return durations.astype(int)

    def act(
        self,
        state: np.ndarray,
        *,
        total_time: int,
        min_green: int,
        deterministic: bool,
    ) -> ActionOutput:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        actor_out, value = self.model(state_t)

        # Convert actor outputs to Dirichlet concentration parameters (alphas > 0).
        alphas = F.softplus(actor_out).squeeze(0) + 1.0
        dist = torch.distributions.Dirichlet(alphas)

        if deterministic:
            proportions = alphas / alphas.sum()
            log_prob = None
            entropy = None
        else:
            proportions = dist.rsample()  # reparameterized sample
            log_prob = dist.log_prob(proportions)
            entropy = dist.entropy()

        durations = self._proportions_to_durations(
            proportions,
            total_time=total_time,
            min_green=min_green,
        )

        return ActionOutput(
            proportions=proportions,
            durations=durations,
            log_prob=log_prob,
            value=value.squeeze(0),
            entropy=entropy,
        )

    def update(
        self,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropies: Optional[torch.Tensor] = None,
    ) -> float:
        advantages = returns - values

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()

        entropy_bonus = torch.tensor(0.0, device=self.device)
        if entropies is not None:
            entropy_bonus = entropies.mean()

        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.detach().cpu().item())

    def save(self, model_path: str) -> None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)

    def load(self, model_path: str, *, map_location: Optional[torch.device] = None) -> None:
        state_dict = torch.load(model_path, map_location=map_location)
        self.model.load_state_dict(state_dict)
        self.model.eval()
