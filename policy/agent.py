from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Independent


class RunningMeanStd:
    """Running mean/std para normalizar observaciones de forma estable."""

    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]
        batch_mean = arr.mean(axis=0)
        batch_var = arr.var(axis=0)
        batch_count = arr.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + float(batch_count)

        new_mean = self.mean + delta * float(batch_count) / total_count
        m_a = self.var * self.count
        m_b = batch_var * float(batch_count)
        m2 = m_a + m_b + np.square(delta) * self.count * float(batch_count) / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-8)
        self.count = total_count

    def normalize(self, x: np.ndarray, clip: float = 5.0) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        norm = (arr - self.mean.astype(np.float32)) / np.sqrt(self.var.astype(np.float32) + 1e-8)
        return np.clip(norm, -clip, clip)

    def state_dict(self) -> dict:
        return {
            "mean": torch.as_tensor(self.mean, dtype=torch.float32),
            "var": torch.as_tensor(self.var, dtype=torch.float32),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64)
        self.var = np.asarray(state["var"], dtype=np.float64)
        self.count = float(state["count"])


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def _inv_softplus(x: float) -> float:
    x = float(max(x, 1e-6))
    return float(np.log(np.expm1(x)))


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        concentration_floor: float = 1.0,
        init_concentration: float = 2.0,
    ):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be > 0, got {obs_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}")
        if concentration_floor < 0.0:
            raise ValueError("concentration_floor must be >= 0")
        if init_concentration <= concentration_floor:
            raise ValueError("init_concentration must be > concentration_floor")

        self.concentration_floor = float(concentration_floor)

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.alpha_head = nn.Linear(hidden_dim, action_dim)
        self.beta_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

        self.backbone.apply(lambda m: _orthogonal_init(m, gain=np.sqrt(2.0)))
        _orthogonal_init(self.value_head, gain=1.0)

        # Arranque simétrico alrededor de 0.5 para evitar sesgo inicial.
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.beta_head.weight)
        init_bias = _inv_softplus(init_concentration - self.concentration_floor)
        nn.init.constant_(self.alpha_head.bias, init_bias)
        nn.init.constant_(self.beta_head.bias, init_bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(obs)
        alpha = F.softplus(self.alpha_head(x)) + self.concentration_floor
        beta = F.softplus(self.beta_head(x)) + self.concentration_floor
        value = self.value_head(x).squeeze(-1)
        params = torch.stack([alpha, beta], dim=-1)
        return params, value


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutBuffer:
    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
    ) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def __len__(self) -> int:
        return len(self.rewards)


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.02,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        minibatch_size: int = 64,
        device: Optional[torch.device] = None,
        normalize_obs: bool = True,
        concentration_floor: float = 1.0,
        init_concentration: float = 2.0,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.normalize_obs = bool(normalize_obs)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.model = ActorCritic(
            obs_dim,
            action_dim,
            hidden_dim=hidden_dim,
            concentration_floor=concentration_floor,
            init_concentration=init_concentration,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.buffer = RolloutBuffer()
        self.obs_rms = RunningMeanStd((self.obs_dim,))

    def _raw_obs(self, obs: np.ndarray) -> np.ndarray:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.shape[0] != self.obs_dim:
            raise ValueError(
                f"Observation dim mismatch: got {arr.shape[0]}, expected {self.obs_dim}."
            )
        return arr

    def prepare_state(self, obs: np.ndarray) -> np.ndarray:
        arr = self._raw_obs(obs)
        if self.normalize_obs:
            arr = self.obs_rms.normalize(arr)
        return arr

    def update_obs_rms(self, obs: np.ndarray) -> None:
        if not self.normalize_obs:
            return
        arr = self._raw_obs(obs)
        self.obs_rms.update(arr[None, :])

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        update_rms: bool = False,
    ) -> Tuple[np.ndarray, float, float, np.ndarray]:
        """
        Devuelve:
        - action
        - log_prob
        - value
        - prepared_state (el estado exacto que vio la red)
        """
        if update_rms:
            self.update_obs_rms(state)

        prepared_state = self.prepare_state(state)
        state_t = torch.as_tensor(prepared_state, dtype=torch.float32, device=self.device).unsqueeze(0)

        params, value = self.model(state_t)
        alpha = params[..., 0].squeeze(0)
        beta = params[..., 1].squeeze(0)
        dist = Independent(Beta(alpha, beta), 1)

        if deterministic:
            action = alpha / (alpha + beta)
        else:
            action = dist.sample()

        action = torch.clamp(action, 1e-6, 1.0 - 1e-6)
        log_prob = dist.log_prob(action)

        return (
            action.cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
            prepared_state.copy(),
        )

    @torch.no_grad()
    def value(self, state: np.ndarray, *, prepared: bool = False) -> float:
        prepared_state = np.asarray(state, dtype=np.float32).reshape(-1) if prepared else self.prepare_state(state)
        state_t = torch.as_tensor(prepared_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, v = self.model(state_t)
        return float(v.item())

    def _compute_returns_and_advantages(self, last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        dones = np.asarray(self.buffer.dones, dtype=np.bool_)
        values = np.asarray(self.buffer.values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        next_value = float(last_value)

        for t in reversed(range(len(rewards))):
            nonterminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[t] = gae
            next_value = values[t]

        returns = advantages + values
        return returns, advantages

    def _iter_minibatches(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ):
        batch_size = states.shape[0]
        indices = torch.randperm(batch_size, device=self.device)
        for start in range(0, batch_size, self.minibatch_size):
            mb_idx = indices[start:start + self.minibatch_size]
            yield RolloutBatch(
                states=states[mb_idx],
                actions=actions[mb_idx],
                old_log_probs=old_log_probs[mb_idx],
                returns=returns[mb_idx],
                advantages=advantages[mb_idx],
            )

    def update(self, last_state: np.ndarray) -> dict:
        if len(self.buffer) == 0:
            return {"updated": False}

        last_state_prepared = self.prepare_state(last_state)
        last_value = self.value(last_state_prepared, prepared=True)

        returns_np, advantages_np = self._compute_returns_and_advantages(last_value)
        advantages_np = (advantages_np - advantages_np.mean()) / (advantages_np.std() + 1e-8)

        states = torch.as_tensor(np.asarray(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(self.buffer.actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(self.buffer.log_probs), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)

        policy_losses = []
        value_losses = []
        entropies = []

        self.model.train()
        for _ in range(self.ppo_epochs):
            for batch in self._iter_minibatches(states, actions, old_log_probs, returns, advantages):
                params, values = self.model(batch.states)
                alpha = params[..., 0]
                beta = params[..., 1]
                dist = Independent(Beta(alpha, beta), 1)

                clipped_actions = torch.clamp(batch.actions, 1e-6, 1.0 - 1e-6)
                new_log_probs = dist.log_prob(clipped_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch.old_log_probs)
                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch.advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, batch.returns)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        self.model.eval()
        self.buffer.clear()

        return {
            "updated": True,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else None,
            "value_loss": float(np.mean(value_losses)) if value_losses else None,
            "entropy": float(np.mean(entropies)) if entropies else None,
        }

    def save(self, path: str, *, metadata: Optional[dict] = None) -> None:
        payload = {
            "model_state_dict": self.model.state_dict(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "metadata": metadata or {},
            "normalize_obs": self.normalize_obs,
            "obs_rms": self.obs_rms.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str, map_location: Optional[torch.device] = None) -> dict:
        load_device = map_location or self.device

        try:
            ckpt = torch.load(path, map_location=load_device, weights_only=True)
        except Exception:
            ckpt = torch.load(path, map_location=load_device, weights_only=False)

        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(
                f"Checkpoint at {path} is incompatible with the current PPO model. "
                "Please retrain with --policy-train."
            )

        ckpt_obs = ckpt.get("obs_dim")
        ckpt_act = ckpt.get("action_dim")

        if ckpt_obs is not None and int(ckpt_obs) != self.obs_dim:
            raise ValueError(
                f"Checkpoint obs_dim={ckpt_obs} does not match current obs_dim={self.obs_dim}. "
                "Please retrain with --policy-train."
            )

        if ckpt_act is not None and int(ckpt_act) != self.action_dim:
            raise ValueError(
                f"Checkpoint action_dim={ckpt_act} does not match current action_dim={self.action_dim}. "
                "Please retrain with --policy-train."
            )

        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)

        ckpt_normalize_obs = ckpt.get("normalize_obs")
        if ckpt_normalize_obs is not None:
            self.normalize_obs = bool(ckpt_normalize_obs)

        if self.normalize_obs and isinstance(ckpt.get("obs_rms"), dict):
            self.obs_rms.load_state_dict(ckpt["obs_rms"])

        self.model.eval()
        return ckpt.get("metadata", {})
