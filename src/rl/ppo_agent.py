# ppo_agent.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Utilities
# ------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PPOConfig:
    state_dim: int
    nvec: np.ndarray                 # MultiDiscrete sizes, shape (D,)
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    epochs: int = 5
    minibatch_size: int = 256
    seed: Optional[int] = 42


# ------------------------------
# Policy: MultiDiscrete (factorized)
# ------------------------------

class MultiDiscretePolicy(nn.Module):
    """
    Factorized policy for gymnasium.spaces.MultiDiscrete.
    Produces a categorical distribution per dimension and sums log-probs.
    """

    def __init__(self, state_dim: int, nvec: np.ndarray):
        super().__init__()
        self.state_dim = int(state_dim)
        self.nvec = np.asarray(nvec, dtype=int)
        self.D = int(self.nvec.shape[0])

        hidden = 256
        self.backbone = nn.Sequential(
            nn.Linear(self.state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Actor heads: one linear per dimension
        self.actor_heads = nn.ModuleList([nn.Linear(hidden, int(n)) for n in self.nvec])
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[list[torch.distributions.Categorical], torch.Tensor]:
        h = self.backbone(x)
        dists = [torch.distributions.Categorical(logits=head(h)) for head in self.actor_heads]
        value = self.critic(h).squeeze(-1)
        return dists, value

    @torch.no_grad()
    def act(self, x: torch.Tensor) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        dists, value = self.forward(x)
        actions = [dist.sample() for dist in dists]
        logprobs = torch.stack([dist.log_prob(a) for dist, a in zip(dists, actions)], dim=-1).sum(dim=-1)
        a_np = torch.stack(actions, dim=-1).cpu().numpy().astype(np.int64)
        return a_np, logprobs, value

    def evaluate(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (sum_logprob, value, sum_entropy). actions shape: (B, D)"""
        dists, value = self.forward(x)
        logps, ents = [], []
        for i, dist in enumerate(dists):
            ai = actions[:, i]
            logps.append(dist.log_prob(ai))
            ents.append(dist.entropy())
        logp = torch.stack(logps, dim=-1).sum(dim=-1)
        ent = torch.stack(ents, dim=-1).sum(dim=-1)
        return logp, value, ent


# ------------------------------
# PPO Agent
# ------------------------------

class PPOAgent:
    def __init__(self, cfg: PPOConfig):
        if cfg.seed is not None:
            np.random.seed(cfg.seed)
            torch.manual_seed(cfg.seed)
        self.device = get_device()
        self.cfg = cfg
        self.policy = MultiDiscretePolicy(cfg.state_dim, cfg.nvec).to(self.device)

        # Separate optimizers
        self.opt_actor = torch.optim.Adam(
            list(self.policy.backbone.parameters()) + list(self.policy.actor_heads.parameters()),
            lr=cfg.lr_actor
        )
        self.opt_critic = torch.optim.Adam(self.policy.critic.parameters(), lr=cfg.lr_critic)

        # Rollout storage
        self._reset_storage()

    def _reset_storage(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []

    @torch.no_grad()
    def select_action(self, state_np: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_np, logp, v = self.policy.act(state_t)
        self.states.append(state_np.astype(np.float32, copy=False))
        self.actions.append(a_np[0].astype(np.int64, copy=False))
        self.logprobs.append(logp.cpu().numpy().astype(np.float32, copy=False))
        self.values.append(v.cpu().numpy().astype(np.float32, copy=False))
        return a_np[0]

    def store_outcome(self, reward: float, done: bool):
        self.rewards.append(float(reward))
        self.dones.append(bool(done))

    def _compute_gae(self, last_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generalized Advantage Estimation with proper handling of dones at every step.
        """
        cfg = self.cfg
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.bool_)
        values = np.asarray(self.values, dtype=np.float32).reshape(-1)

        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + cfg.gamma * next_value * nonterminal - values[t]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            adv[t] = gae
        returns = adv + values

        advantages = torch.as_tensor(adv, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        return advantages, returns_t

    def update(self):
        cfg = self.cfg
        states = torch.as_tensor(np.asarray(self.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(self.actions), dtype=torch.int64, device=self.device)
        old_logp = torch.as_tensor(np.asarray(self.logprobs).reshape(-1), dtype=torch.float32, device=self.device)

        # last value for GAE bootstrap
        with torch.no_grad():
            last_state = states[-1:].detach()
            _, last_value = self.policy.forward(last_state)

        adv, ret = self._compute_gae(last_value=float(last_value.squeeze().cpu().numpy()))
        # normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # SGD over minibatches
        B = states.shape[0]
        idx = np.arange(B)
        for _ in range(cfg.epochs):
            np.random.shuffle(idx)
            for start in range(0, B, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                s = states[mb]
                a = actions[mb]
                olp = old_logp[mb]
                adv_b = adv[mb]
                ret_b = ret[mb]

                # -------- Actor step (uses shared backbone) --------
                logp, value, ent = self.policy.evaluate(s, a)  # value unused here but harmless
                ratio = torch.exp(logp - olp)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean() - cfg.ent_coef * ent.mean()

                self.opt_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt_actor.step()

                # -------- Critic step: RECOMPUTE forward to get a fresh graph --------
                with torch.enable_grad():
                    # forward again to avoid second backward through freed graph
                    dists, value2 = self.policy.forward(s)  # only need value
                    critic_loss = F.mse_loss(value2, ret_b)

                self.opt_critic.zero_grad(set_to_none=True)
                (cfg.vf_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt_critic.step()

        self._reset_storage()

    def save(self, path: str):
        torch.save({
            "policy": self.policy.state_dict(),
            "cfg": self.cfg.__dict__,
        }, path)

    def load(self, path: str, map_location: Optional[str] = None):
        ckpt = torch.load(path, map_location=map_location or get_device())
        self.policy.load_state_dict(ckpt["policy"])
