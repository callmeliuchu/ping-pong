from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from pingpong_rl.envs.league_self_play_env import LeagueSelfPlayConfig, LeagueSelfPlayEnv
from train.eval_utils import ROOT, print_metrics, write_metrics


def _zip_exists(path: Path) -> bool:
    return path.exists() or path.with_suffix(".zip").exists()


def default_opponent_paths() -> list[Path]:
    candidates = [
        ROOT / "models" / "passed" / "ppo_stage10",
        ROOT / "models" / "passed" / "ppo_stage11",
        ROOT / "models" / "selfplay" / "stage12" / "gen_1",
        ROOT / "models" / "passed" / "ppo_stage12",
        ROOT / "models" / "passed" / "ppo_stage13_red",
    ]
    paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and _zip_exists(candidate):
            paths.append(candidate)
            seen.add(resolved)
    return paths


def _label(path: Path) -> str:
    if path.parent.name == "stage12":
        return path.stem
    if path.parent.name == "passed":
        return path.stem
    return path.with_suffix("").name


def _elo_delta_from_win_rate(win_rate: float) -> float:
    win_rate = min(max(win_rate, 0.01), 0.99)
    return -400.0 * math.log10((1.0 / win_rate) - 1.0)


def _evaluate_one(model: PPO, opponent_model_path: Path, episodes: int, seed: int) -> dict[str, Any]:
    config = LeagueSelfPlayConfig(opponent_model_paths=(str(opponent_model_path),))
    env = LeagueSelfPlayEnv(config=config)
    totals = {
        "terminated": 0,
        "truncated": 0,
        "scores": 0,
        "misses": 0,
        "hit_episodes": 0,
        "agent_hits": 0,
        "opponent_hits": 0,
        "rally": 0,
        "reward": 0.0,
        "steps": 0,
        "loop_landing_episodes": 0,
        "drive_landing_episodes": 0,
        "smash_landing_episodes": 0,
        "high_arc_episodes": 0,
        "opponent_loaded": 0,
    }
    point_reasons: dict[str, int] = {}

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        info: dict[str, Any] = {}
        terminated = False
        truncated = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            done = terminated or truncated

        totals["terminated"] += int(terminated)
        totals["truncated"] += int(truncated)
        totals["scores"] += int(info["agent_score"])
        totals["misses"] += int(info["agent_miss"])
        totals["hit_episodes"] += int(info["agent_hits"] > 0)
        totals["agent_hits"] += int(info["agent_hits"])
        totals["opponent_hits"] += int(info["opponent_hits"])
        totals["rally"] += int(info["rally_length"])
        totals["reward"] += episode_reward
        totals["steps"] += int(info["steps"])
        totals["loop_landing_episodes"] += int(info["loop_landings"] > 0)
        totals["drive_landing_episodes"] += int(info["drive_landings"] > 0)
        totals["smash_landing_episodes"] += int(info["smash_landings"] > 0)
        totals["high_arc_episodes"] += int(info["high_arc_landings"] > 0)
        totals["opponent_loaded"] += int(info["opponent_model_loaded"])
        reason = str(info.get("point_reason", "unknown"))
        point_reasons[reason] = point_reasons.get(reason, 0) + 1

    env.close()
    win_rate = totals["scores"] / episodes
    return {
        "episodes": episodes,
        "normal_end_rate": totals["terminated"] / episodes,
        "truncated_rate": totals["truncated"] / episodes,
        "win_rate": win_rate,
        "miss_rate": totals["misses"] / episodes,
        "hit_rate": totals["hit_episodes"] / episodes,
        "avg_agent_hits": totals["agent_hits"] / episodes,
        "avg_opponent_hits": totals["opponent_hits"] / episodes,
        "avg_rally_length": totals["rally"] / episodes,
        "loop_landing_rate": totals["loop_landing_episodes"] / episodes,
        "drive_landing_rate": totals["drive_landing_episodes"] / episodes,
        "smash_landing_rate": totals["smash_landing_episodes"] / episodes,
        "high_arc_rate": totals["high_arc_episodes"] / episodes,
        "opponent_loaded_rate": totals["opponent_loaded"] / episodes,
        "estimated_elo_delta": _elo_delta_from_win_rate(win_rate),
        "point_reasons": point_reasons,
        "avg_reward": totals["reward"] / episodes,
        "avg_steps": totals["steps"] / episodes,
    }


def evaluate_league_model(
    model_path: Path,
    opponent_model_paths: list[Path],
    episodes: int,
    seed: int = 0,
) -> dict[str, Any]:
    model = PPO.load(model_path)
    per_opponent: dict[str, dict[str, Any]] = {}
    for index, opponent_path in enumerate(opponent_model_paths):
        per_opponent[_label(opponent_path)] = _evaluate_one(model, opponent_path, episodes, seed + index * 10_000)

    opponent_count = len(per_opponent)
    if opponent_count == 0:
        raise ValueError("Stage 13 league evaluation needs at least one opponent model.")

    def avg(key: str) -> float:
        return sum(float(metrics[key]) for metrics in per_opponent.values()) / opponent_count

    win_rates = {name: float(metrics["win_rate"]) for name, metrics in per_opponent.items()}
    elo_delta = sum(float(metrics["estimated_elo_delta"]) for metrics in per_opponent.values()) / opponent_count
    return {
        "stage": 13,
        "episodes_per_opponent": episodes,
        "opponent_pool_size": opponent_count,
        "pool_win_rate": avg("win_rate"),
        "worst_opponent_win_rate": min(win_rates.values()),
        "best_opponent_win_rate": max(win_rates.values()),
        "normal_end_rate": avg("normal_end_rate"),
        "truncated_rate": avg("truncated_rate"),
        "hit_rate": avg("hit_rate"),
        "avg_agent_hits": avg("avg_agent_hits"),
        "avg_opponent_hits": avg("avg_opponent_hits"),
        "avg_rally_length": avg("avg_rally_length"),
        "loop_landing_rate": avg("loop_landing_rate"),
        "drive_landing_rate": avg("drive_landing_rate"),
        "smash_landing_rate": avg("smash_landing_rate"),
        "high_arc_rate": avg("high_arc_rate"),
        "opponent_loaded_rate": avg("opponent_loaded_rate"),
        "estimated_elo": 1000.0 + elo_delta,
        "estimated_elo_delta_vs_pool": elo_delta,
        "win_rate_matrix": {"candidate": win_rates},
        "elo_ratings": {"candidate": 1000.0 + elo_delta, **{name: 1000.0 for name in per_opponent}},
        "per_opponent": per_opponent,
        "avg_reward": avg("avg_reward"),
        "avg_steps": avg("avg_steps"),
    }


def evaluate_red_challenge(
    red_model_path: Path,
    blue_model_path: Path,
    episodes: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate a model loaded as the red-side mirrored opponent against a blue champion."""

    blue_model = PPO.load(blue_model_path)
    config = LeagueSelfPlayConfig(opponent_model_paths=(str(red_model_path),))
    env = LeagueSelfPlayEnv(config=config)
    totals = {
        "terminated": 0,
        "truncated": 0,
        "red_scores": 0,
        "blue_scores": 0,
        "blue_hits": 0,
        "red_hits": 0,
        "hit_episodes": 0,
        "rally": 0,
        "reward": 0.0,
        "steps": 0,
        "opponent_loaded": 0,
    }
    point_reasons: dict[str, int] = {}

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        info: dict[str, Any] = {}
        terminated = False
        truncated = False
        while not done:
            action, _ = blue_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            done = terminated or truncated

        totals["terminated"] += int(terminated)
        totals["truncated"] += int(truncated)
        totals["red_scores"] += int(info["agent_miss"])
        totals["blue_scores"] += int(info["agent_score"])
        totals["blue_hits"] += int(info["agent_hits"])
        totals["red_hits"] += int(info["opponent_hits"])
        totals["hit_episodes"] += int(info["opponent_hits"] > 0)
        totals["rally"] += int(info["rally_length"])
        totals["reward"] += episode_reward
        totals["steps"] += int(info["steps"])
        totals["opponent_loaded"] += int(info["opponent_model_loaded"])
        reason = str(info.get("point_reason", "unknown"))
        point_reasons[reason] = point_reasons.get(reason, 0) + 1

    env.close()
    red_win_rate = totals["red_scores"] / episodes
    blue_win_rate = totals["blue_scores"] / episodes
    return {
        "stage": 13,
        "episodes": episodes,
        "red_model_path": str(red_model_path),
        "blue_model_path": str(blue_model_path),
        "normal_end_rate": totals["terminated"] / episodes,
        "truncated_rate": totals["truncated"] / episodes,
        "red_win_rate": red_win_rate,
        "blue_win_rate": blue_win_rate,
        "red_hit_rate": totals["hit_episodes"] / episodes,
        "avg_red_hits": totals["red_hits"] / episodes,
        "avg_blue_hits": totals["blue_hits"] / episodes,
        "avg_rally_length": totals["rally"] / episodes,
        "opponent_loaded_rate": totals["opponent_loaded"] / episodes,
        "estimated_red_elo_delta": _elo_delta_from_win_rate(red_win_rate),
        "point_reasons": point_reasons,
        "avg_blue_reward": totals["reward"] / episodes,
        "avg_steps": totals["steps"] / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 13 league self-play against an opponent pool.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage13")
    parser.add_argument("--opponent-model-paths", type=Path, nargs="*", default=None)
    parser.add_argument("--red-challenge-model-path", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-path", type=Path, default=None)
    args = parser.parse_args()

    if args.red_challenge_model_path is not None:
        metrics = evaluate_red_challenge(args.red_challenge_model_path, args.model_path, args.episodes, seed=args.seed)
    else:
        opponent_paths = args.opponent_model_paths if args.opponent_model_paths is not None else default_opponent_paths()
        metrics = evaluate_league_model(args.model_path, opponent_paths, args.episodes, seed=args.seed)
    write_metrics(metrics, args.json_path)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
