from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from train.eval_utils import ROOT, write_metrics
from train.evaluate_robot_arm_league import default_robot_arm_opponent_paths, evaluate_robot_arm_league_model
from train.train_robot_arm_league import _zip_source, main as train_main


def _score(metrics: dict[str, Any]) -> float:
    return (
        100.0 * float(metrics["pool_win_rate"])
        + 40.0 * float(metrics["worst_opponent_win_rate"])
        + 5.0 * float(metrics["avg_rally_length"])
        + 12.0 * float(metrics["loop_landing_rate"])
        + 10.0 * float(metrics["drive_landing_rate"])
        + 0.04 * float(metrics["estimated_elo_delta_vs_pool"])
    )


def should_promote(
    candidate: dict[str, Any],
    champion: dict[str, Any],
    min_score_improvement: float,
    min_pool_win_rate: float,
    min_worst_win_rate: float,
    min_rally_length: float,
) -> tuple[bool, str]:
    candidate_score = _score(candidate)
    champion_score = _score(champion)
    if float(candidate["pool_win_rate"]) < min_pool_win_rate:
        return False, f"pool_win_rate {candidate['pool_win_rate']:.3f} < {min_pool_win_rate:.3f}"
    if float(candidate["worst_opponent_win_rate"]) < min_worst_win_rate:
        return False, f"worst_opponent_win_rate {candidate['worst_opponent_win_rate']:.3f} < {min_worst_win_rate:.3f}"
    if float(candidate["avg_rally_length"]) < min_rally_length:
        return False, f"avg_rally_length {candidate['avg_rally_length']:.2f} < {min_rally_length:.2f}"
    if float(candidate["loop_landing_rate"]) < 0.80 or float(candidate["drive_landing_rate"]) < 0.80:
        return False, "loop_landing_rate and drive_landing_rate must both stay >= 0.80"
    if candidate_score < champion_score + min_score_improvement:
        return False, f"score {candidate_score:.2f} < champion score {champion_score:.2f} + {min_score_improvement:.2f}"
    return True, f"score {candidate_score:.2f} >= champion score {champion_score:.2f} + {min_score_improvement:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve Stage 15 robot-arm league and promote stronger candidates.")
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--champion-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage15")
    parser.add_argument("--base-model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage14")
    parser.add_argument("--candidate-path", type=Path, default=ROOT / "models" / "ppo_robot_arm_league_stage15")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--min-score-improvement", type=float, default=0.25)
    parser.add_argument("--min-pool-win-rate", type=float, default=0.45)
    parser.add_argument("--min-worst-win-rate", type=float, default=0.35)
    parser.add_argument("--min-rally-length", type=float, default=8.0)
    parser.add_argument("--history-path", type=Path, default=ROOT / "logs" / "ppo_robot_arm_league_stage15" / "evolution_history.json")
    args = parser.parse_args()

    initial_champion = args.champion_path if _zip_source(args.champion_path).exists() else args.base_model_path
    opponent_paths = default_robot_arm_opponent_paths()
    champion_metrics = evaluate_robot_arm_league_model(initial_champion, opponent_paths, args.eval_episodes, seed=args.seed)
    write_metrics(champion_metrics, ROOT / "logs" / "ppo_robot_arm_league_stage15" / f"champion_before_gen{args.generation}.json")

    import sys

    previous_argv = sys.argv
    try:
        sys.argv = [
            "train_robot_arm_league",
            "--generation",
            str(args.generation),
            "--timesteps",
            str(args.timesteps),
            "--num-envs",
            str(args.num_envs),
            "--seed",
            str(args.seed + args.generation),
            "--base-model-path",
            str(initial_champion),
            "--model-path",
            str(args.candidate_path),
            "--eval-episodes",
            str(args.eval_episodes),
        ]
        train_main()
    finally:
        sys.argv = previous_argv

    candidate_metrics = evaluate_robot_arm_league_model(args.candidate_path, opponent_paths, args.eval_episodes, seed=args.seed)
    write_metrics(candidate_metrics, ROOT / "logs" / "ppo_robot_arm_league_stage15" / f"gen{args.generation}_evolve_eval.json")

    promoted, reason = should_promote(
        candidate_metrics,
        champion_metrics,
        args.min_score_improvement,
        args.min_pool_win_rate,
        args.min_worst_win_rate,
        args.min_rally_length,
    )
    if promoted:
        args.champion_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_zip_source(args.candidate_path), args.champion_path if args.champion_path.suffix == ".zip" else args.champion_path.with_suffix(".zip"))

    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if args.history_path.exists():
        history = json.loads(args.history_path.read_text())
    history.append(
        {
            "generation": args.generation,
            "promoted": promoted,
            "reason": reason,
            "champion_score": _score(champion_metrics),
            "candidate_score": _score(candidate_metrics),
            "champion": champion_metrics,
            "candidate": candidate_metrics,
        }
    )
    args.history_path.write_text(json.dumps(history, indent=2, sort_keys=True))
    print(f"promoted={promoted} reason={reason}")


if __name__ == "__main__":
    main()
