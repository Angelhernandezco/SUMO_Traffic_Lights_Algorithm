from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Independent


class ActorCritic(nn.Module):
	def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
		super().__init__()
		if obs_dim <= 0:
			raise ValueError(f"obs_dim must be > 0, got {obs_dim}")
		if action_dim <= 0:
			raise ValueError(f"action_dim must be > 0, got {action_dim}")

		self.backbone = nn.Sequential(
			nn.Linear(obs_dim, hidden_dim),
			nn.Tanh(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.Tanh(),
		)
		# Output parameters for independent Beta distributions per action dimension.
		self.alpha_head = nn.Linear(hidden_dim, action_dim)
		self.beta_head = nn.Linear(hidden_dim, action_dim)
		self.value_head = nn.Linear(hidden_dim, 1)

	def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		x = self.backbone(obs)
		# Beta parameters must be strictly positive.
		alpha = F.softplus(self.alpha_head(x)) + 1e-3
		beta = F.softplus(self.beta_head(x)) + 1e-3
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
		lr: float = 3e-4,
		gamma: float = 0.99,
		gae_lambda: float = 0.95,
		clip_eps: float = 0.2,
		entropy_coef: float = 0.01,
		value_coef: float = 0.5,
		max_grad_norm: float = 0.5,
		ppo_epochs: int = 10,
		minibatch_size: int = 64,
		device: Optional[torch.device] = None,
	) -> None:
		self.obs_dim = obs_dim
		self.action_dim = action_dim

		self.gamma = float(gamma)
		self.gae_lambda = float(gae_lambda)
		self.clip_eps = float(clip_eps)
		self.entropy_coef = float(entropy_coef)
		self.value_coef = float(value_coef)
		self.max_grad_norm = float(max_grad_norm)
		self.ppo_epochs = int(ppo_epochs)
		self.minibatch_size = int(minibatch_size)

		if device is None:
			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.device = device

		self.model = ActorCritic(obs_dim, action_dim).to(self.device)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

		self.buffer = RolloutBuffer()

	@torch.no_grad()
	def act(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
		state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
		params, value = self.model(state_t)
		alpha = params[..., 0].squeeze(0)
		beta = params[..., 1].squeeze(0)
		dist = Independent(Beta(alpha, beta), 1)
		action = dist.sample()
		log_prob = dist.log_prob(action)
		return action.cpu().numpy(), float(log_prob.item()), float(value.item())

	@torch.no_grad()
	def value(self, state: np.ndarray) -> float:
		state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
		_, v = self.model(state_t)
		return float(v.item())

	def _compute_returns_and_advantages(
		self, last_value: float
	) -> Tuple[np.ndarray, np.ndarray]:
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
			mb_idx = indices[start : start + self.minibatch_size]
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

		last_value = self.value(last_state)
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

		for _ in range(self.ppo_epochs):
			for batch in self._iter_minibatches(states, actions, old_log_probs, returns, advantages):
				params, values = self.model(batch.states)
				alpha = params[..., 0]
				beta = params[..., 1]
				dist = Independent(Beta(alpha, beta), 1)
				new_log_probs = dist.log_prob(batch.actions)
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
		}
		torch.save(payload, path)

	def load(self, path: str, map_location: Optional[torch.device] = None) -> dict:
		ckpt = torch.load(path, map_location=map_location or self.device)
		# Backward compatibility: allow raw state_dict.
		if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
			ckpt_obs = ckpt.get("obs_dim")
			ckpt_act = ckpt.get("action_dim")
			if ckpt_obs is not None and int(ckpt_obs) != int(self.obs_dim):
				raise ValueError(
					f"Checkpoint obs_dim={ckpt_obs} does not match current obs_dim={self.obs_dim}."
				)
			if ckpt_act is not None and int(ckpt_act) != int(self.action_dim):
				raise ValueError(
					f"Checkpoint action_dim={ckpt_act} does not match current action_dim={self.action_dim}."
				)

			self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
			self.model.eval()
			return ckpt.get("metadata", {})
		if isinstance(ckpt, dict):
			# Unknown format: attempt strict load; if it fails, raise a helpful error.
			try:
				self.model.load_state_dict(ckpt, strict=True)
			except Exception as exc:  # noqa: BLE001
				raise ValueError(
					f"Checkpoint at {path} is incompatible with the current PPO model. "
					f"Please retrain with --policy-train. Original error: {exc}"
				) from exc
			self.model.eval()
			return {}
		raise ValueError(f"Unsupported checkpoint format in {path}")

