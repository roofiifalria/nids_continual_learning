from __future__ import annotations
from typing import Dict
from .ppo_agent import PPOAgent
import time
import numpy as np


def train_with_ppo(
    env,
    agent: PPOAgent,
    total_steps: int = 50000,
    rollout_len: int = 2048,
    log_every: int = 1000,
    eval_every: int = 5000,
) -> Dict[str, list]:
    """
    Minimal PPO training loop for continual-learning NIDS environments.

    - env: Gym environment (Hybrid, GAN, or Real)
    - agent: PPOAgent with MultiDiscrete action head
    - total_steps: Total environment interaction steps
    - rollout_len: Number of steps per PPO update
    - log_every: Print progress every ~log_every steps
    - eval_every: Frequency for evaluation logging (kept for future extensions)
    """
    logs = {"step": [], "reward": [], "acc": [], "f1": []}
    obs, _ = env.reset()
    step, ep_return = 0, 0.0
    start_time = time.time()
    recent_returns = []

    while step < total_steps:
        rollout_start = step
        # Collect rollout
        for _ in range(rollout_len):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_outcome(reward, done)
            ep_return += reward
            obs = next_obs
            step += 1

            if done:
                logs["step"].append(step)
                logs["reward"].append(ep_return)
                if isinstance(info, dict) and "metrics" in info:
                    logs["acc"].append(info["metrics"].get("acc", np.nan))
                    logs["f1"].append(info["metrics"].get("f1", np.nan))
                recent_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()

            if step >= total_steps:
                break

        # PPO update
        if len(agent.rewards) >= agent.cfg.minibatch_size:
            agent.update()

        # Logging heartbeat
        if step % log_every < rollout_len:
            try:
                m = info.get("metrics", {})
                avg_ret = np.mean(recent_returns[-10:]) if recent_returns else 0.0
                print(
                    f"[PPO] step={step:>6d} | "
                    f"avgR={avg_ret:>7.4f} | "
                    f"acc={m.get('acc', float('nan')):>6.3f} | "
                    f"f1={m.get('f1', float('nan')):>6.3f}"
                )
            except Exception:
                pass

    print(f"\n[Training completed in {(time.time() - start_time)/60:.2f} min]")

    # Convert lists to arrays for downstream plotting
    for k in logs:
        logs[k] = np.asarray(logs[k], dtype=np.float32)

    return logs
