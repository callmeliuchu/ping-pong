from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from pingpong_rl.envs.robot_arm_tactical_league_env import RobotArmTacticalLeagueConfig, RobotArmTacticalLeagueEnv
from train.eval_utils import ROOT, print_metrics, write_metrics
from train.evaluate_robot_arm_league import default_robot_arm_opponent_paths


def _label(path: Path) -> str:
    if path.parent.name == "passed":
        return path.stem
    if path.parent.name in {"stage15", "stage16"}:
        return path.stem
    return path.with_suffix("").name


def _elo_delta_from_win_rate(win_rate: float) -> float:
    win_rate = min(max(win_rate, 0.01), 0.99)
    return -400.0 * math.log10((1.0 / win_rate) - 1.0)


def default_tactical_opponent_paths() -> list[Path]:
    candidates = default_robot_arm_opponent_paths()
    candidates.extend(sorted((ROOT / "models" / "selfplay" / "stage16").glob("gen_*.zip")))
    paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        source = candidate if candidate.suffix == ".zip" else candidate.with_suffix(".zip")
        if not source.exists():
            continue
        resolved = source.resolve()
        if resolved in seen:
            continue
        paths.append(candidate)
        seen.add(resolved)
    return paths


def _evaluate_one(model: PPO, opponent_model_path: Path, episodes: int, seed: int) -> dict[str, Any]:
    config = RobotArmTacticalLeagueConfig(opponent_model_paths=(str(opponent_model_path),))
    env = RobotArmTacticalLeagueEnv(config=config)
    totals = {
        "terminated": 0,
        "truncated": 0,
        "scores": 0,
        "misses": 0,
        "hit_episodes": 0,
        "agent_hits": 0,
        "opponent_hits": 0,
        "rally": 0,
        "legal_landings": 0,
        "reward": 0.0,
        "steps": 0,
        "loop_episodes": 0,
        "drive_episodes": 0,
        "topspin_episodes": 0,
        "max_topspin": 0.0,
        "tracking_error": 0.0,
        "opponent_loaded": 0,
        "red_drive_episodes": 0,
        "clean_scores": 0,
        "wrong_side_scores": 0,
    }
    point_reasons: dict[str, int] = {}

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        tracking_total = 0.0
        tracking_samples = 0
        info: dict[str, Any] = {}
        terminated = False
        truncated = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            tracking_total += float(info.get("agent_tracking_error", 0.0))
            tracking_samples += 1
            done = terminated or truncated

        totals["terminated"] += int(terminated)
        totals["truncated"] += int(truncated)
        totals["scores"] += int(info["agent_score"])
        totals["misses"] += int(info["agent_miss"])
        totals["hit_episodes"] += int(info["agent_hits"] > 0)
        totals["agent_hits"] += int(info["agent_hits"])
        totals["opponent_hits"] += int(info["opponent_hits"])
        totals["rally"] += int(info["rally_length"])
        totals["legal_landings"] += int(info["legal_landings"])
        totals["reward"] += episode_reward
        totals["steps"] += int(info["steps"])
        totals["loop_episodes"] += int(info.get("loop_landings", 0) > 0)
        totals["drive_episodes"] += int(info.get("drive_landings", 0) > 0)
        totals["topspin_episodes"] += int(info.get("topspin_landings", 0) > 0)
        totals["max_topspin"] += float(info.get("max_topspin", 0.0))
        totals["tracking_error"] += tracking_total / max(tracking_samples, 1)
        totals["opponent_loaded"] += int(info.get("opponent_model_loaded", False))
        totals["red_drive_episodes"] += int(info.get("red_drive_attempts", 0) > 0)
        totals["clean_scores"] += int(info.get("clean_agent_scores", 0))
        totals["wrong_side_scores"] += int(info.get("wrong_side_agent_scores", 0))
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
        "avg_legal_landings": totals["legal_landings"] / episodes,
        "loop_landing_rate": totals["loop_episodes"] / episodes,
        "drive_landing_rate": totals["drive_episodes"] / episodes,
        "topspin_landing_rate": totals["topspin_episodes"] / episodes,
        "avg_max_topspin": totals["max_topspin"] / episodes,
        "avg_tracking_error": totals["tracking_error"] / episodes,
        "opponent_loaded_rate": totals["opponent_loaded"] / episodes,
        "red_drive_attempt_rate": totals["red_drive_episodes"] / episodes,
        "clean_score_rate": totals["clean_scores"] / episodes,
        "wrong_side_score_rate": totals["wrong_side_scores"] / episodes,
        "estimated_elo_delta": _elo_delta_from_win_rate(win_rate),
        "point_reasons": point_reasons,
        "avg_reward": totals["reward"] / episodes,
        "avg_steps": totals["steps"] / episodes,
    }


def evaluate_robot_arm_tactical_model(
    model_path: Path,
    opponent_model_paths: list[Path],
    episodes: int,
    seed: int = 0,
) -> dict[str, Any]:
    model = PPO.load(model_path)
    per_opponent: dict[str, dict[str, Any]] = {}
    for index, opponent_path in enumerate(opponent_model_paths):
        per_opponent[_label(opponent_path)] = _evaluate_one(model, opponent_path, episodes, seed + index * 10_000)
    if not per_opponent:
        raise ValueError("Stage 16 tactical evaluation needs at least one robot-arm opponent model.")

    def avg(key: str) -> float:
        return sum(float(metrics[key]) for metrics in per_opponent.values()) / len(per_opponent)

    win_rates = {name: float(metrics["win_rate"]) for name, metrics in per_opponent.items()}
    elo_delta = sum(float(metrics["estimated_elo_delta"]) for metrics in per_opponent.values()) / len(per_opponent)
    return {
        "stage": 16,
        "episodes_per_opponent": episodes,
        "opponent_pool_size": len(per_opponent),
        "pool_win_rate": avg("win_rate"),
        "worst_opponent_win_rate": min(win_rates.values()),
        "best_opponent_win_rate": max(win_rates.values()),
        "normal_end_rate": avg("normal_end_rate"),
        "truncated_rate": avg("truncated_rate"),
        "hit_rate": avg("hit_rate"),
        "avg_agent_hits": avg("avg_agent_hits"),
        "avg_opponent_hits": avg("avg_opponent_hits"),
        "avg_rally_length": avg("avg_rally_length"),
        "avg_legal_landings": avg("avg_legal_landings"),
        "loop_landing_rate": avg("loop_landing_rate"),
        "drive_landing_rate": avg("drive_landing_rate"),
        "topspin_landing_rate": avg("topspin_landing_rate"),
        "avg_max_topspin": avg("avg_max_topspin"),
        "avg_tracking_error": avg("avg_tracking_error"),
        "opponent_loaded_rate": avg("opponent_loaded_rate"),
        "red_drive_attempt_rate": avg("red_drive_attempt_rate"),
        "clean_score_rate": avg("clean_score_rate"),
        "wrong_side_score_rate": avg("wrong_side_score_rate"),
        "estimated_elo": 1000.0 + elo_delta,
        "estimated_elo_delta_vs_pool": elo_delta,
        "win_rate_matrix": {"candidate": win_rates},
        "elo_ratings": {"candidate": 1000.0 + elo_delta, **{name: 1000.0 for name in per_opponent}},
        "per_opponent": per_opponent,
        "avg_reward": avg("avg_reward"),
        "avg_steps": avg("avg_steps"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 16 tactical robot-arm league.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage16")
    parser.add_argument("--opponent-model-paths", type=Path, nargs="*", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-path", type=Path, default=None)
    args = parser.parse_args()

    opponent_paths = args.opponent_model_paths if args.opponent_model_paths is not None else default_tactical_opponent_paths()
    metrics = evaluate_robot_arm_tactical_model(args.model_path, opponent_paths, args.episodes, seed=args.seed)
    write_metrics(metrics, args.json_path)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
