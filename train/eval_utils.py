from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from pingpong_rl.envs import CatchEnv, GravityPingPongEnv, PongEnv, make_pong_config


ROOT = Path(__file__).resolve().parents[1]


def write_metrics(metrics: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_stage1(
    model_path: Path = ROOT / "models" / "ppo_catch_stage1",
    episodes: int = 100,
    render: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    env = CatchEnv(render_mode="human" if render else None)
    model = PPO.load(model_path)
    hits = 0
    misses = 0
    total_reward = 0.0
    total_steps = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            done = terminated or truncated
        hits += int(info["agent_hit"])
        misses += int(info["agent_miss"])
        total_reward += episode_reward
        total_steps += int(info["steps"])

    env.close()
    return {
        "stage": 1,
        "episodes": episodes,
        "hit_rate": hits / episodes,
        "miss_rate": misses / episodes,
        "avg_reward": total_reward / episodes,
        "avg_steps": total_steps / episodes,
    }


def evaluate_pong(
    stage: int,
    model_path: Path | None = None,
    opponent_model_path: str | None = None,
    episodes: int = 100,
    render: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    resolved_model_path = model_path or ROOT / "models" / f"ppo_pong_stage{stage}"
    env = PongEnv(
        render_mode="human" if render else None,
        config=make_pong_config(stage, opponent_model_path),
    )
    model = PPO.load(resolved_model_path)

    wins = 0
    misses = 0
    final_step_hits = 0
    episodes_with_hit = 0
    total_hits = 0
    total_agent_hits = 0
    total_opponent_hits = 0
    total_rally = 0
    total_reward = 0.0
    total_rally = 0
    total_steps = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        final_info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, final_info = env.step(int(action))
            episode_reward += reward
            done = terminated or truncated

        wins += int(final_info["agent_score"])
        misses += int(final_info["agent_miss"])
        final_step_hits += int(final_info["agent_hit"])
        episodes_with_hit += int(final_info["agent_hits"] > 0)
        total_hits += int(final_info["agent_hits"])
        total_reward += episode_reward
        total_rally += int(final_info["rally_length"])
        total_steps += int(final_info["steps"])

    env.close()
    return {
        "stage": stage,
        "episodes": episodes,
        "win_rate": wins / episodes,
        "miss_rate": misses / episodes,
        "hit_rate": episodes_with_hit / episodes,
        "episode_hit_rate": episodes_with_hit / episodes,
        "avg_agent_hits": total_hits / episodes,
        "final_step_hit_rate": final_step_hits / episodes,
        "avg_reward": total_reward / episodes,
        "avg_rally_length": total_rally / episodes,
        "avg_steps": total_steps / episodes,
    }


def smoke_gravity(episodes: int = 100, render: bool = False, seed: int = 0) -> dict[str, Any]:
    env = GravityPingPongEnv(render_mode="human" if render else None)
    terminated_count = 0
    truncated_count = 0
    total_hits = 0
    total_agent_hits = 0
    total_opponent_hits = 0
    total_rally = 0
    total_scores = 0
    total_misses = 0
    total_steps = 0
    max_steps_seen = 0

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        info = {}
        terminated = False
        truncated = False
        while not done:
            action = 0
            if env.ball_vx < 0:
                target_y = env.predict_ball_y_at_x(env.config.paddle_x)
            else:
                target_y = env.config.table_y - 90
            if target_y < env.agent_y - 5:
                action = 1
            elif target_y > env.agent_y + 5:
                action = 2
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        terminated_count += int(terminated)
        truncated_count += int(truncated)
        total_hits += int(info["agent_hit"])
        total_agent_hits += int(info["agent_hits"])
        total_opponent_hits += int(info["opponent_hits"])
        total_rally += int(info["rally_length"])
        total_scores += int(info["agent_score"])
        total_misses += int(info["agent_miss"])
        total_steps += int(info["steps"])
        max_steps_seen = max(max_steps_seen, int(info["steps"]))

    env.close()
    return {
        "stage": 6,
        "episodes": episodes,
        "normal_end_rate": terminated_count / episodes,
        "truncated_rate": truncated_count / episodes,
        "final_step_hit_rate": total_hits / episodes,
        "avg_agent_hits": total_agent_hits / episodes,
        "avg_opponent_hits": total_opponent_hits / episodes,
        "avg_rally_length": total_rally / episodes,
        "score_rate": total_scores / episodes,
        "miss_rate": total_misses / episodes,
        "avg_steps": total_steps / episodes,
        "max_steps_seen": max_steps_seen,
    }


def print_metrics(metrics: dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}={value:.3f}")
        else:
            print(f"{key}={value}")
