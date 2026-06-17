from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO

from pingpong_rl.envs.robot_arm_bilateral_league_env import RobotArmBilateralLeagueConfig, RobotArmBilateralLeagueEnv
from train.eval_utils import ROOT, print_metrics, write_metrics
from train.evaluate_robot_arm_grand_champion_attack import default_grand_champion_attack_opponent_paths


def _zip_path(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _label(path: Path) -> str:
    if path.parent.name == "passed":
        return path.stem
    if path.parent.name in {"stage15", "stage16", "stage17", "stage18", "stage19", "stage20", "stage21", "stage22"}:
        return f"{path.parent.name}_{path.stem}"
    return path.with_suffix("").name


def _elo_delta_from_win_rate(win_rate: float) -> float:
    win_rate = min(max(win_rate, 0.01), 0.99)
    return -400.0 * math.log10((1.0 / win_rate) - 1.0)


def _unique_existing(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        source = _zip_path(path)
        if not source.exists():
            continue
        resolved = source.resolve()
        if resolved in seen:
            continue
        unique.append(path)
        seen.add(resolved)
    return unique


def default_bilateral_opponent_paths() -> list[Path]:
    candidates = default_grand_champion_attack_opponent_paths()
    candidates.extend(
        [
            ROOT / "models" / "passed" / "ppo_stage21",
            ROOT / "models" / "passed" / "ppo_stage22_blue",
            ROOT / "models" / "passed" / "ppo_stage22_red",
        ]
    )
    candidates.extend(sorted((ROOT / "models" / "selfplay" / "stage22").glob("blue_gen_*.zip")))
    candidates.extend(sorted((ROOT / "models" / "selfplay" / "stage22").glob("red_gen_*.zip")))
    return _unique_existing(candidates)


def _evaluate_blue_one(model: PPO, opponent_model_path: Path, profile_index: int, episodes: int, seed: int) -> dict[str, Any]:
    config = RobotArmBilateralLeagueConfig(
        opponent_model_paths=(str(opponent_model_path),),
        pressure_profile_index=profile_index,
    )
    env = RobotArmBilateralLeagueEnv(config=config)
    totals = {
        "terminated": 0,
        "truncated": 0,
        "blue_scores": 0,
        "red_scores": 0,
        "hit_episodes": 0,
        "blue_hits": 0,
        "red_hits": 0,
        "rally": 0,
        "legal_landings": 0,
        "reward": 0.0,
        "steps": 0,
        "loop_episodes": 0,
        "drive_episodes": 0,
        "topspin_episodes": 0,
        "opponent_loaded": 0,
        "forced_finish_scores": 0,
        "timely_finish_scores": 0,
        "stalemate_finishes": 0,
        "combo_attack_episodes": 0,
        "wrong_side_scores": 0,
    }
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
        totals["blue_scores"] += int(info["agent_score"])
        totals["red_scores"] += int(info["agent_miss"])
        totals["hit_episodes"] += int(info["agent_hits"] > 0)
        totals["blue_hits"] += int(info["agent_hits"])
        totals["red_hits"] += int(info["opponent_hits"])
        totals["rally"] += int(info["rally_length"])
        totals["legal_landings"] += int(info["legal_landings"])
        totals["reward"] += episode_reward
        totals["steps"] += int(info["steps"])
        totals["loop_episodes"] += int(info.get("loop_landings", 0) > 0)
        totals["drive_episodes"] += int(info.get("drive_landings", 0) > 0)
        totals["topspin_episodes"] += int(info.get("topspin_landings", 0) > 0)
        totals["opponent_loaded"] += int(info.get("opponent_model_loaded", False))
        totals["forced_finish_scores"] += int(info.get("forced_finish_scores", 0))
        totals["timely_finish_scores"] += int(info.get("timely_finish_scores", 0))
        totals["stalemate_finishes"] += int(info.get("stalemate_finishes", 0))
        totals["combo_attack_episodes"] += int(info.get("combo_attack_landings", 0) > 0)
        totals["wrong_side_scores"] += int(info.get("wrong_side_agent_scores", 0))

    env.close()
    blue_win_rate = totals["blue_scores"] / episodes
    return {
        "profile": profile_name,
        "episodes": episodes,
        "normal_end_rate": totals["terminated"] / episodes,
        "truncated_rate": totals["truncated"] / episodes,
        "blue_win_rate": blue_win_rate,
        "red_win_rate": totals["red_scores"] / episodes,
        "hit_rate": totals["hit_episodes"] / episodes,
        "avg_blue_hits": totals["blue_hits"] / episodes,
        "avg_red_hits": totals["red_hits"] / episodes,
        "avg_rally_length": totals["rally"] / episodes,
        "avg_legal_landings": totals["legal_landings"] / episodes,
        "loop_landing_rate": totals["loop_episodes"] / episodes,
        "drive_landing_rate": totals["drive_episodes"] / episodes,
        "topspin_landing_rate": totals["topspin_episodes"] / episodes,
        "opponent_loaded_rate": totals["opponent_loaded"] / episodes,
        "forced_finish_rate": totals["forced_finish_scores"] / episodes,
        "timely_finish_rate": totals["timely_finish_scores"] / episodes,
        "stalemate_rate": totals["stalemate_finishes"] / episodes,
        "combo_attack_rate": totals["combo_attack_episodes"] / episodes,
        "wrong_side_score_rate": totals["wrong_side_scores"] / episodes,
        "estimated_elo_delta": _elo_delta_from_win_rate(blue_win_rate),
        "avg_reward": totals["reward"] / episodes,
        "avg_steps": totals["steps"] / episodes,
    }


def evaluate_blue_bilateral_model(
    model_path: Path,
    opponent_model_paths: list[Path],
    episodes: int,
    seed: int = 0,
) -> dict[str, Any]:
    model = PPO.load(model_path)
    profiles = RobotArmBilateralLeagueConfig().pressure_profiles
    per_match: dict[str, dict[str, Any]] = {}
    for profile_index, profile in enumerate(profiles):
        for opponent_index, opponent_path in enumerate(opponent_model_paths):
            name = f"{profile[0]}:{_label(opponent_path)}"
            per_match[name] = _evaluate_blue_one(
                model,
                opponent_path,
                profile_index,
                episodes,
                seed + profile_index * 100_000 + opponent_index * 10_000,
            )
    if not per_match:
        raise ValueError("Stage 22 bilateral evaluation needs at least one opponent model.")

    def avg(key: str) -> float:
        return sum(float(metrics[key]) for metrics in per_match.values()) / len(per_match)

    def profile_avg(profile_name: str, key: str) -> float:
        matches = [metrics for metrics in per_match.values() if metrics["profile"] == profile_name]
        return sum(float(metrics[key]) for metrics in matches) / len(matches)

    win_rates = {name: float(metrics["blue_win_rate"]) for name, metrics in per_match.items()}
    profile_win_rates = {profile[0]: profile_avg(profile[0], "blue_win_rate") for profile in profiles}
    elo_delta = sum(float(metrics["estimated_elo_delta"]) for metrics in per_match.values()) / len(per_match)
    return {
        "stage": 22,
        "controlled_side": "blue",
        "episodes_per_matchup": episodes,
        "opponent_pool_size": len(opponent_model_paths),
        "profile_count": len(profiles),
        "matchup_count": len(per_match),
        "pool_win_rate": avg("blue_win_rate"),
        "worst_matchup_win_rate": min(win_rates.values()),
        "min_profile_win_rate": min(profile_win_rates.values()),
        "profile_win_rates": profile_win_rates,
        "normal_end_rate": avg("normal_end_rate"),
        "hit_rate": avg("hit_rate"),
        "avg_rally_length": avg("avg_rally_length"),
        "avg_legal_landings": avg("avg_legal_landings"),
        "loop_landing_rate": avg("loop_landing_rate"),
        "drive_landing_rate": avg("drive_landing_rate"),
        "topspin_landing_rate": avg("topspin_landing_rate"),
        "opponent_loaded_rate": avg("opponent_loaded_rate"),
        "forced_finish_rate": avg("forced_finish_rate"),
        "timely_finish_rate": avg("timely_finish_rate"),
        "stalemate_rate": avg("stalemate_rate"),
        "combo_attack_rate": avg("combo_attack_rate"),
        "wrong_side_score_rate": avg("wrong_side_score_rate"),
        "estimated_elo": 1000.0 + elo_delta,
        "estimated_elo_delta_vs_pool": elo_delta,
        "win_rate_matrix": {"blue_candidate": win_rates},
        "per_matchup": per_match,
        "avg_reward": avg("avg_reward"),
        "avg_steps": avg("avg_steps"),
    }


def evaluate_red_challenge(
    red_model_path: Path,
    blue_model_path: Path,
    episodes: int,
    seed: int = 0,
) -> dict[str, Any]:
    blue_model = PPO.load(blue_model_path)
    profiles = RobotArmBilateralLeagueConfig().pressure_profiles
    per_profile: dict[str, dict[str, Any]] = {}
    for profile_index, profile in enumerate(profiles):
        metrics = _evaluate_blue_one(
            blue_model,
            red_model_path,
            profile_index,
            episodes,
            seed + profile_index * 100_000,
        )
        per_profile[profile[0]] = metrics

    def avg(key: str) -> float:
        return sum(float(metrics[key]) for metrics in per_profile.values()) / len(per_profile)

    red_win_rate = avg("red_win_rate")
    return {
        "stage": 22,
        "controlled_side": "red_challenge",
        "red_model_path": str(red_model_path),
        "blue_model_path": str(blue_model_path),
        "episodes_per_profile": episodes,
        "profile_count": len(profiles),
        "red_win_rate": red_win_rate,
        "blue_win_rate": avg("blue_win_rate"),
        "red_hit_rate": avg("hit_rate"),
        "avg_rally_length": avg("avg_rally_length"),
        "avg_legal_landings": avg("avg_legal_landings"),
        "normal_end_rate": avg("normal_end_rate"),
        "red_estimated_elo_delta": _elo_delta_from_win_rate(red_win_rate),
        "blue_forced_finish_rate": avg("forced_finish_rate"),
        "blue_stalemate_rate": avg("stalemate_rate"),
        "per_profile": per_profile,
    }


def bilateral_score(metrics: dict[str, Any]) -> float:
    return (
        100.0 * float(metrics["pool_win_rate"])
        + 45.0 * float(metrics["worst_matchup_win_rate"])
        + 4.0 * float(metrics["avg_rally_length"])
        + 28.0 * float(metrics["forced_finish_rate"])
        - 25.0 * float(metrics["stalemate_rate"])
        + 0.04 * float(metrics["estimated_elo_delta_vs_pool"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage 22 bilateral robot-arm league.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage22")
    parser.add_argument("--opponent-model-paths", type=Path, nargs="*", default=None)
    parser.add_argument("--red-challenge-model-path", type=Path, default=None)
    parser.add_argument("--blue-model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage22_blue")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-path", type=Path, default=None)
    args = parser.parse_args()

    if args.red_challenge_model_path is not None:
        metrics = evaluate_red_challenge(args.red_challenge_model_path, args.blue_model_path, args.episodes, seed=args.seed)
    else:
        opponent_paths = args.opponent_model_paths if args.opponent_model_paths is not None else default_bilateral_opponent_paths()
        metrics = evaluate_blue_bilateral_model(args.model_path, opponent_paths, args.episodes, seed=args.seed)
        metrics["bilateral_score"] = bilateral_score(metrics)
    write_metrics(metrics, args.json_path)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
