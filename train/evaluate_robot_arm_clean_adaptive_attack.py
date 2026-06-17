from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from pingpong_rl.envs.robot_arm_clean_adaptive_attack_env import (
    RobotArmCleanAdaptiveAttackConfig,
    RobotArmCleanAdaptiveAttackEnv,
)
from train.eval_utils import ROOT, print_metrics, write_metrics
from train.evaluate_robot_arm_adaptive_attack import default_adaptive_opponent_paths


def _label(path: Path) -> str:
    if path.parent.name == "passed":
        return path.stem
    if path.parent.name in {"stage15", "stage16", "stage17", "stage18", "stage19"}:
        return f"{path.parent.name}_{path.stem}"
    return path.with_suffix("").name


def _elo_delta_from_win_rate(win_rate: float) -> float:
    win_rate = min(max(win_rate, 0.01), 0.99)
    return -400.0 * math.log10((1.0 / win_rate) - 1.0)


def default_clean_adaptive_opponent_paths() -> list[Path]:
    candidates = default_adaptive_opponent_paths()
    candidates.extend(sorted((ROOT / "models" / "selfplay" / "stage19").glob("gen_*.zip")))
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


def _evaluate_one(model: PPO, opponent_model_path: Path, profile_index: int, episodes: int, seed: int) -> dict[str, Any]:
    config = RobotArmCleanAdaptiveAttackConfig(
        opponent_model_paths=(str(opponent_model_path),),
        pressure_profile_index=profile_index,
    )
    env = RobotArmCleanAdaptiveAttackEnv(config=config)
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
        "opponent_loaded": 0,
        "wrong_side_scores": 0,
        "attack_episodes": 0,
        "high_pressure_episodes": 0,
        "clean_attack_scores": 0,
        "adaptive_clean_scores": 0,
        "clean_combo_scores": 0,
        "deep_attack_episodes": 0,
        "wide_attack_episodes": 0,
        "combo_attack_episodes": 0,
        "max_attack_pressure": 0.0,
    }
    point_reasons: dict[str, int] = {}
    profile_name = config.pressure_profiles[profile_index][0]

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
        totals["legal_landings"] += int(info["legal_landings"])
        totals["reward"] += episode_reward
        totals["steps"] += int(info["steps"])
        totals["loop_episodes"] += int(info.get("loop_landings", 0) > 0)
        totals["drive_episodes"] += int(info.get("drive_landings", 0) > 0)
        totals["topspin_episodes"] += int(info.get("topspin_landings", 0) > 0)
        totals["max_topspin"] += float(info.get("max_topspin", 0.0))
        totals["opponent_loaded"] += int(info.get("opponent_model_loaded", False))
        totals["wrong_side_scores"] += int(info.get("wrong_side_agent_scores", 0))
        totals["attack_episodes"] += int(info.get("attack_landings", 0) > 0)
        totals["high_pressure_episodes"] += int(info.get("high_pressure_landings", 0) > 0)
        totals["clean_attack_scores"] += int(info.get("clean_attack_scores", 0))
        totals["adaptive_clean_scores"] += int(info.get("adaptive_clean_scores", 0))
        totals["clean_combo_scores"] += int(info.get("clean_combo_scores", 0))
        totals["deep_attack_episodes"] += int(info.get("deep_attack_landings", 0) > 0)
        totals["wide_attack_episodes"] += int(info.get("wide_attack_landings", 0) > 0)
        totals["combo_attack_episodes"] += int(info.get("combo_attack_landings", 0) > 0)
        totals["max_attack_pressure"] += float(info.get("max_attack_pressure", 0.0))
        reason = str(info.get("point_reason", "unknown"))
        point_reasons[reason] = point_reasons.get(reason, 0) + 1

    env.close()
    win_rate = totals["scores"] / episodes
    return {
        "profile": profile_name,
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
        "opponent_loaded_rate": totals["opponent_loaded"] / episodes,
        "wrong_side_score_rate": totals["wrong_side_scores"] / episodes,
        "attack_landing_rate": totals["attack_episodes"] / episodes,
        "high_pressure_rate": totals["high_pressure_episodes"] / episodes,
        "clean_attack_score_rate": totals["clean_attack_scores"] / episodes,
        "adaptive_clean_score_rate": totals["adaptive_clean_scores"] / episodes,
        "clean_combo_score_rate": totals["clean_combo_scores"] / episodes,
        "deep_attack_rate": totals["deep_attack_episodes"] / episodes,
        "wide_attack_rate": totals["wide_attack_episodes"] / episodes,
        "combo_attack_rate": totals["combo_attack_episodes"] / episodes,
        "avg_max_attack_pressure": totals["max_attack_pressure"] / episodes,
        "estimated_elo_delta": _elo_delta_from_win_rate(win_rate),
        "point_reasons": point_reasons,
        "avg_reward": totals["reward"] / episodes,
        "avg_steps": totals["steps"] / episodes,
    }


def evaluate_robot_arm_clean_adaptive_attack_model(
    model_path: Path,
    opponent_model_paths: list[Path],
    episodes: int,
    seed: int = 0,
) -> dict[str, Any]:
    model = PPO.load(model_path)
    profiles = RobotArmCleanAdaptiveAttackConfig().pressure_profiles
    per_match: dict[str, dict[str, Any]] = {}
    for profile_index, profile in enumerate(profiles):
        for opponent_index, opponent_path in enumerate(opponent_model_paths):
            name = f"{profile[0]}:{_label(opponent_path)}"
            per_match[name] = _evaluate_one(
                model,
                opponent_path,
                profile_index,
                episodes,
                seed + profile_index * 100_000 + opponent_index * 10_000,
            )
    if not per_match:
        raise ValueError("Stage 19 clean adaptive attack evaluation needs at least one robot-arm opponent model.")

    def avg(key: str) -> float:
        return sum(float(metrics[key]) for metrics in per_match.values()) / len(per_match)

    def profile_avg(profile_name: str, key: str) -> float:
        matches = [metrics for metrics in per_match.values() if metrics["profile"] == profile_name]
        return sum(float(metrics[key]) for metrics in matches) / len(matches)

    win_rates = {name: float(metrics["win_rate"]) for name, metrics in per_match.items()}
    profile_win_rates = {profile[0]: profile_avg(profile[0], "win_rate") for profile in profiles}
    elo_delta = sum(float(metrics["estimated_elo_delta"]) for metrics in per_match.values()) / len(per_match)
    return {
        "stage": 19,
        "episodes_per_matchup": episodes,
        "opponent_pool_size": len(opponent_model_paths),
        "profile_count": len(profiles),
        "matchup_count": len(per_match),
        "pool_win_rate": avg("win_rate"),
        "worst_matchup_win_rate": min(win_rates.values()),
        "min_profile_win_rate": min(profile_win_rates.values()),
        "profile_win_rates": profile_win_rates,
        "normal_end_rate": avg("normal_end_rate"),
        "truncated_rate": avg("truncated_rate"),
        "hit_rate": avg("hit_rate"),
        "avg_rally_length": avg("avg_rally_length"),
        "avg_legal_landings": avg("avg_legal_landings"),
        "loop_landing_rate": avg("loop_landing_rate"),
        "drive_landing_rate": avg("drive_landing_rate"),
        "topspin_landing_rate": avg("topspin_landing_rate"),
        "avg_max_topspin": avg("avg_max_topspin"),
        "opponent_loaded_rate": avg("opponent_loaded_rate"),
        "wrong_side_score_rate": avg("wrong_side_score_rate"),
        "attack_landing_rate": avg("attack_landing_rate"),
        "high_pressure_rate": avg("high_pressure_rate"),
        "clean_attack_score_rate": avg("clean_attack_score_rate"),
        "adaptive_clean_score_rate": avg("adaptive_clean_score_rate"),
        "clean_combo_score_rate": avg("clean_combo_score_rate"),
        "deep_attack_rate": avg("deep_attack_rate"),
        "wide_attack_rate": avg("wide_attack_rate"),
        "combo_attack_rate": avg("combo_attack_rate"),
        "avg_max_attack_pressure": avg("avg_max_attack_pressure"),
        "estimated_elo": 1000.0 + elo_delta,
        "estimated_elo_delta_vs_pool": elo_delta,
        "win_rate_matrix": {"candidate": win_rates},
        "per_matchup": per_match,
        "avg_reward": avg("avg_reward"),
        "avg_steps": avg("avg_steps"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 19 clean adaptive robot-arm league.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage19")
    parser.add_argument("--opponent-model-paths", type=Path, nargs="*", default=None)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-path", type=Path, default=None)
    args = parser.parse_args()

    opponent_paths = args.opponent_model_paths if args.opponent_model_paths is not None else default_clean_adaptive_opponent_paths()
    metrics = evaluate_robot_arm_clean_adaptive_attack_model(args.model_path, opponent_paths, args.episodes, seed=args.seed)
    write_metrics(metrics, args.json_path)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
