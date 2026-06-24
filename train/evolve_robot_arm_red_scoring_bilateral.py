from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pingpong_rl.envs.robot_arm_red_scoring_bilateral_env import (
    RedRobotArmRedScoringBilateralEnv,
    RobotArmRedScoringBilateralConfig,
    RobotArmRedScoringBilateralEnv,
)
from train.eval_utils import ROOT, write_metrics
from train.evaluate_robot_arm_red_scoring_bilateral import (
    default_red_scoring_bilateral_opponent_paths,
    evaluate_blue_red_scoring_bilateral_model,
    evaluate_red_scoring_challenge,
    red_scoring_bilateral_score,
)


def _zip_path(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _copy_model(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_zip_path(source), _zip_path(destination))


def _ensure_initial_champions(blue_path: Path, red_path: Path, fallback_blue_path: Path, fallback_red_path: Path) -> None:
    if not _zip_path(blue_path).exists():
        _copy_model(fallback_blue_path, blue_path)
    if not _zip_path(red_path).exists():
        _copy_model(fallback_red_path, red_path)


def _make_blue_env(opponent_paths: list[Path]):
    def _init():
        config = RobotArmRedScoringBilateralConfig(opponent_model_paths=tuple(str(path) for path in opponent_paths))
        return Monitor(RobotArmRedScoringBilateralEnv(render_mode=None, config=config))

    return _init


def _make_red_env(blue_paths: list[Path]):
    def _init():
        config = RobotArmRedScoringBilateralConfig(blue_model_paths=tuple(str(path) for path in blue_paths))
        return Monitor(RedRobotArmRedScoringBilateralEnv(render_mode=None, config=config))

    return _init


def _train_candidate(
    train_side: str,
    base_model_path: Path,
    candidate_path: Path,
    generation: int,
    timesteps: int,
    num_envs: int,
    seed: int,
    eval_episodes: int,
    opponent_paths: list[Path],
    blue_paths: list[Path],
) -> Path:
    if train_side == "red":
        env_fns = [_make_red_env(blue_paths) for _ in range(num_envs)]
        eval_env = Monitor(
            RedRobotArmRedScoringBilateralEnv(
                config=RobotArmRedScoringBilateralConfig(blue_model_paths=tuple(str(path) for path in blue_paths))
            )
        )
    else:
        env_fns = [_make_blue_env(opponent_paths) for _ in range(num_envs)]
        eval_env = Monitor(
            RobotArmRedScoringBilateralEnv(
                config=RobotArmRedScoringBilateralConfig(
                    opponent_model_paths=tuple(str(path) for path in opponent_paths)
                )
            )
        )

    vec_env = DummyVecEnv(env_fns)
    vec_env.seed(seed)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(timesteps // 4 // num_envs, 1),
        save_path=str(ROOT / "models" / "checkpoints"),
        name_prefix=f"ppo_robot_arm_red_scoring_bilateral_stage23_{train_side}_gen{generation}",
    )
    best_dir = ROOT / "models" / "best" / f"stage23_{train_side}_gen{generation}"
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(ROOT / "logs" / "ppo_robot_arm_red_scoring_bilateral_stage23" / f"eval_{train_side}_gen{generation}"),
        eval_freq=max(timesteps // 8 // num_envs, 1),
        n_eval_episodes=min(eval_episodes, 50),
        deterministic=True,
        render=False,
    )

    model = PPO.load(base_model_path, env=vec_env)
    model.verbose = 1
    model.learn(total_timesteps=timesteps, callback=CallbackList([checkpoint_callback, eval_callback]))
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(candidate_path)
    vec_env.close()
    eval_env.close()
    best_model = best_dir / "best_model"
    return best_model if _zip_path(best_model).exists() else candidate_path


def _blue_promote(candidate: dict[str, Any], champion: dict[str, Any], min_improvement: float) -> tuple[bool, str]:
    candidate_score = red_scoring_bilateral_score(candidate)
    champion_score = red_scoring_bilateral_score(champion)
    if float(candidate["pool_win_rate"]) < 0.60:
        return False, f"blue pool_win_rate {candidate['pool_win_rate']:.3f} < 0.600"
    if float(candidate["worst_matchup_win_rate"]) < 0.35:
        return False, f"blue worst_matchup_win_rate {candidate['worst_matchup_win_rate']:.3f} < 0.350"
    if float(candidate["forced_finish_rate"]) < 0.30:
        return False, f"blue forced_finish_rate {candidate['forced_finish_rate']:.3f} < 0.300"
    if float(candidate["loop_landing_rate"]) < 0.85:
        return False, f"blue loop_landing_rate {candidate['loop_landing_rate']:.3f} < 0.850"
    if float(candidate["topspin_landing_rate"]) < 0.85:
        return False, f"blue topspin_landing_rate {candidate['topspin_landing_rate']:.3f} < 0.850"
    if candidate_score < champion_score + min_improvement:
        return False, f"blue score {candidate_score:.2f} < champion {champion_score:.2f} + {min_improvement:.2f}"
    return True, f"blue score {candidate_score:.2f} >= champion {champion_score:.2f} + {min_improvement:.2f}"


def _red_promote(candidate: dict[str, Any], incumbent: dict[str, Any], min_improvement: float) -> tuple[bool, str]:
    candidate_win = float(candidate["red_win_rate"])
    incumbent_win = float(incumbent["red_win_rate"])
    if float(candidate["normal_end_rate"]) < 0.95:
        return False, f"red normal_end_rate {candidate['normal_end_rate']:.3f} < 0.950"
    if float(candidate["red_hit_rate"]) < 0.90:
        return False, f"red_hit_rate {candidate['red_hit_rate']:.3f} < 0.900"
    if float(candidate["avg_rally_length"]) < 4.5:
        return False, f"red avg_rally_length {candidate['avg_rally_length']:.2f} < 4.50"
    if candidate_win < 0.08 and candidate_win < incumbent_win + min_improvement:
        return (
            False,
            f"red_win_rate {candidate_win:.3f} < 0.080 and < incumbent {incumbent_win:.3f} + {min_improvement:.3f}",
        )
    return True, f"red_win_rate {candidate_win:.3f} improves incumbent {incumbent_win:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve Stage 23 red-scoring bilateral robot-arm champions.")
    parser.add_argument("--start-generation", type=int, default=1)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--timesteps-per-generation", type=int, default=120_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23001)
    parser.add_argument("--sequence", nargs="*", choices=("blue", "red"), default=("red", "red", "blue", "red", "red", "blue"))
    parser.add_argument("--blue-champion-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage23_blue")
    parser.add_argument("--red-champion-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage23_red")
    parser.add_argument("--fallback-blue-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage22_blue")
    parser.add_argument("--fallback-red-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage22_red")
    parser.add_argument("--stage23-model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage23")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--min-blue-score-improvement", type=float, default=0.50)
    parser.add_argument("--min-red-win-improvement", type=float, default=0.05)
    parser.add_argument("--history-path", type=Path, default=ROOT / "logs" / "ppo_robot_arm_red_scoring_bilateral_stage23" / "history.json")
    parser.add_argument("--skip-check-env", action="store_true")
    args = parser.parse_args()

    _ensure_initial_champions(args.blue_champion_path, args.red_champion_path, args.fallback_blue_path, args.fallback_red_path)
    _copy_model(args.blue_champion_path, args.stage23_model_path)
    if not args.skip_check_env:
        check_env(RobotArmRedScoringBilateralEnv(config=RobotArmRedScoringBilateralConfig()))
        check_env(
            RedRobotArmRedScoringBilateralEnv(
                config=RobotArmRedScoringBilateralConfig(blue_model_paths=(str(args.blue_champion_path),))
            )
        )

    league_dir = ROOT / "models" / "selfplay" / "stage23"
    league_dir.mkdir(parents=True, exist_ok=True)
    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = json.loads(args.history_path.read_text()) if args.history_path.exists() else []

    sequence = list(args.sequence)
    for offset in range(args.generations):
        generation = args.start_generation + offset
        train_side = sequence[offset % len(sequence)]
        opponent_paths = default_red_scoring_bilateral_opponent_paths()
        blue_paths = [args.blue_champion_path]
        base_model_path = args.red_champion_path if train_side == "red" else args.blue_champion_path
        candidate_path = league_dir / f"{train_side}_gen_{generation}"
        best_candidate_path = _train_candidate(
            train_side,
            base_model_path,
            candidate_path,
            generation,
            args.timesteps_per_generation,
            args.num_envs,
            args.seed + generation,
            args.eval_episodes,
            opponent_paths,
            blue_paths,
        )
        if _zip_path(best_candidate_path).resolve() != _zip_path(candidate_path).resolve():
            _copy_model(best_candidate_path, candidate_path)

        record: dict[str, Any] = {
            "generation": generation,
            "train_side": train_side,
            "candidate_path": str(_zip_path(candidate_path)),
            "best_checkpoint_path": str(_zip_path(best_candidate_path)),
            "opponent_pool": [str(path) for path in opponent_paths],
            "blue_champion_path": str(_zip_path(args.blue_champion_path)),
            "red_champion_path": str(_zip_path(args.red_champion_path)),
        }

        if train_side == "red":
            incumbent_metrics = evaluate_red_scoring_challenge(
                args.red_champion_path,
                args.blue_champion_path,
                args.eval_episodes,
                seed=args.seed + generation * 10_000,
            )
            candidate_metrics = evaluate_red_scoring_challenge(
                candidate_path,
                args.blue_champion_path,
                args.eval_episodes,
                seed=args.seed + generation * 10_000,
            )
            promoted, reason = _red_promote(candidate_metrics, incumbent_metrics, args.min_red_win_improvement)
            if promoted:
                _copy_model(candidate_path, args.red_champion_path)
            write_metrics(
                candidate_metrics,
                ROOT / "logs" / "ppo_robot_arm_red_scoring_bilateral_stage23" / f"red_gen{generation}_eval.json",
            )
            record.update(
                {
                    "promoted": promoted,
                    "reason": reason,
                    "incumbent_metrics": incumbent_metrics,
                    "candidate_metrics": candidate_metrics,
                    "red_champion_path_after": str(_zip_path(args.red_champion_path)),
                }
            )
        else:
            champion_metrics = evaluate_blue_red_scoring_bilateral_model(
                args.blue_champion_path,
                opponent_paths,
                args.eval_episodes,
                seed=args.seed + generation * 10_000,
            )
            candidate_metrics = evaluate_blue_red_scoring_bilateral_model(
                candidate_path,
                opponent_paths,
                args.eval_episodes,
                seed=args.seed + generation * 10_000,
            )
            promoted, reason = _blue_promote(candidate_metrics, champion_metrics, args.min_blue_score_improvement)
            if promoted:
                _copy_model(candidate_path, args.blue_champion_path)
                _copy_model(args.blue_champion_path, args.stage23_model_path)
            write_metrics(
                candidate_metrics,
                ROOT / "logs" / "ppo_robot_arm_red_scoring_bilateral_stage23" / f"blue_gen{generation}_eval.json",
            )
            record.update(
                {
                    "promoted": promoted,
                    "reason": reason,
                    "champion_score": red_scoring_bilateral_score(champion_metrics),
                    "candidate_score": red_scoring_bilateral_score(candidate_metrics),
                    "champion_metrics": champion_metrics,
                    "candidate_metrics": candidate_metrics,
                    "blue_champion_path_after": str(_zip_path(args.blue_champion_path)),
                }
            )

        _copy_model(args.blue_champion_path, args.stage23_model_path)
        history.append(record)
        args.history_path.write_text(json.dumps(history, indent=2, sort_keys=True))
        print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
