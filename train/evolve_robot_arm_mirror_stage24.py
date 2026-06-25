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

from pingpong_rl.envs.robot_arm_mirror_stage24_env import RobotArmMirrorSelfPlayConfig, RobotArmMirrorSelfPlayEnv
from train.eval_utils import ROOT, write_metrics
from train.evaluate_robot_arm_mirror_stage24 import (
    default_mirror_stage24_opponent_paths,
    evaluate_mirror_stage24_model,
    mirror_stage24_score,
)


def _zip_path(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _copy_model(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_zip_path(source), _zip_path(destination))


def _ensure_initial_champions(left_path: Path, right_path: Path, stage24_path: Path, fallback_path: Path) -> None:
    if not _zip_path(left_path).exists():
        _copy_model(fallback_path, left_path)
    _copy_model(left_path, right_path)
    _copy_model(left_path, stage24_path)


def _make_env(mirror_path: Path):
    def _init():
        config = RobotArmMirrorSelfPlayConfig(opponent_model_paths=(str(mirror_path),))
        return Monitor(RobotArmMirrorSelfPlayEnv(render_mode=None, config=config))

    return _init


def _train_candidate(
    base_model_path: Path,
    mirror_model_path: Path,
    candidate_path: Path,
    generation: int,
    timesteps: int,
    num_envs: int,
    seed: int,
    eval_episodes: int,
) -> Path:
    env_fns = [_make_env(mirror_model_path) for _ in range(num_envs)]
    eval_env = Monitor(
        RobotArmMirrorSelfPlayEnv(
            config=RobotArmMirrorSelfPlayConfig(opponent_model_paths=(str(mirror_model_path),))
        )
    )
    vec_env = DummyVecEnv(env_fns)
    vec_env.seed(seed)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(timesteps // 4 // num_envs, 1),
        save_path=str(ROOT / "models" / "checkpoints"),
        name_prefix=f"ppo_robot_arm_mirror_stage24_gen{generation}",
    )
    best_dir = ROOT / "models" / "best" / f"stage24_gen{generation}"
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(ROOT / "logs" / "ppo_robot_arm_mirror_stage24" / f"eval_gen{generation}"),
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


def _promote(candidate: dict[str, Any], champion: dict[str, Any], min_score_improvement: float) -> tuple[bool, str]:
    candidate_score = mirror_stage24_score(candidate)
    champion_score = mirror_stage24_score(champion)
    if float(candidate["mirror_win_rate"]) < 0.52:
        return False, f"mirror_win_rate {candidate['mirror_win_rate']:.3f} < 0.520"
    if float(candidate["pool_win_rate"]) < 0.62:
        return False, f"pool_win_rate {candidate['pool_win_rate']:.3f} < 0.620"
    if float(candidate["hit_rate"]) < 0.95:
        return False, f"hit_rate {candidate['hit_rate']:.3f} < 0.950"
    if float(candidate["avg_rally_length"]) < 5.5:
        return False, f"avg_rally_length {candidate['avg_rally_length']:.2f} < 5.50"
    if float(candidate["normal_end_rate"]) < 0.95:
        return False, f"normal_end_rate {candidate['normal_end_rate']:.3f} < 0.950"
    if float(candidate["forced_finish_rate"]) < 0.25:
        return False, f"forced_finish_rate {candidate['forced_finish_rate']:.3f} < 0.250"
    if float(candidate["stalemate_rate"]) > 0.25:
        return False, f"stalemate_rate {candidate['stalemate_rate']:.3f} > 0.250"
    if candidate_score < champion_score + min_score_improvement:
        return False, f"stage24_score {candidate_score:.2f} < champion {champion_score:.2f} + {min_score_improvement:.2f}"
    return True, f"stage24_score {candidate_score:.2f} >= champion {champion_score:.2f} + {min_score_improvement:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolve Stage 24 by copying each left champion to the right mirror.")
    parser.add_argument("--start-generation", type=int, default=1)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--timesteps-per-generation", type=int, default=120_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=24001)
    parser.add_argument("--left-champion-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage24_left")
    parser.add_argument("--right-champion-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage24_right")
    parser.add_argument("--stage24-model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage24")
    parser.add_argument("--fallback-champion-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage23_blue")
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--min-score-improvement", type=float, default=0.50)
    parser.add_argument("--max-stale-generations", type=int, default=3)
    parser.add_argument("--history-path", type=Path, default=ROOT / "logs" / "ppo_robot_arm_mirror_stage24" / "history.json")
    parser.add_argument("--skip-check-env", action="store_true")
    args = parser.parse_args()

    _ensure_initial_champions(
        args.left_champion_path,
        args.right_champion_path,
        args.stage24_model_path,
        args.fallback_champion_path,
    )
    if not args.skip_check_env:
        check_env(RobotArmMirrorSelfPlayEnv(config=RobotArmMirrorSelfPlayConfig(opponent_model_paths=(str(args.right_champion_path),))))

    league_dir = ROOT / "models" / "selfplay" / "stage24"
    league_dir.mkdir(parents=True, exist_ok=True)
    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = json.loads(args.history_path.read_text()) if args.history_path.exists() else []
    stale_generations = 0

    champion_metrics = evaluate_mirror_stage24_model(
        args.left_champion_path,
        args.right_champion_path,
        default_mirror_stage24_opponent_paths(),
        args.eval_episodes,
        seed=args.seed,
    )
    write_metrics(champion_metrics, ROOT / "logs" / "ppo_robot_arm_mirror_stage24" / "champion_baseline_eval.json")

    for offset in range(args.generations):
        generation = args.start_generation + offset
        _copy_model(args.left_champion_path, args.right_champion_path)
        candidate_path = league_dir / f"candidate_gen_{generation}"
        best_candidate_path = _train_candidate(
            args.left_champion_path,
            args.right_champion_path,
            candidate_path,
            generation,
            args.timesteps_per_generation,
            args.num_envs,
            args.seed + generation,
            args.eval_episodes,
        )
        if _zip_path(best_candidate_path).resolve() != _zip_path(candidate_path).resolve():
            _copy_model(best_candidate_path, candidate_path)

        candidate_metrics = evaluate_mirror_stage24_model(
            candidate_path,
            args.right_champion_path,
            default_mirror_stage24_opponent_paths(),
            args.eval_episodes,
            seed=args.seed + generation * 10_000,
        )
        promoted, reason = _promote(candidate_metrics, champion_metrics, args.min_score_improvement)
        champion_path_after = args.left_champion_path
        if promoted:
            _copy_model(candidate_path, args.left_champion_path)
            _copy_model(args.left_champion_path, args.right_champion_path)
            _copy_model(args.left_champion_path, args.stage24_model_path)
            champion_snapshot = league_dir / f"champion_gen_{generation}"
            _copy_model(args.left_champion_path, champion_snapshot)
            champion_path_after = champion_snapshot
            champion_metrics = evaluate_mirror_stage24_model(
                args.left_champion_path,
                args.right_champion_path,
                default_mirror_stage24_opponent_paths(),
                args.eval_episodes,
                seed=args.seed + generation * 10_000 + 5000,
            )
            stale_generations = 0
        else:
            stale_generations += 1

        write_metrics(candidate_metrics, ROOT / "logs" / "ppo_robot_arm_mirror_stage24" / f"candidate_gen{generation}_eval.json")
        record = {
            "generation": generation,
            "candidate_path": str(_zip_path(candidate_path)),
            "best_checkpoint_path": str(_zip_path(best_candidate_path)),
            "promoted": promoted,
            "reason": reason,
            "candidate_score": mirror_stage24_score(candidate_metrics),
            "champion_score": mirror_stage24_score(champion_metrics),
            "candidate_metrics": candidate_metrics,
            "champion_metrics_after": champion_metrics,
            "left_champion_path": str(_zip_path(args.left_champion_path)),
            "right_mirror_path": str(_zip_path(args.right_champion_path)),
            "champion_snapshot_path": str(_zip_path(champion_path_after)),
            "stale_generations": stale_generations,
        }
        history.append(record)
        args.history_path.write_text(json.dumps(history, indent=2, sort_keys=True))
        print(json.dumps(record, indent=2, sort_keys=True))
        if stale_generations >= args.max_stale_generations:
            print(f"Stopping after {stale_generations} non-promoting generations.")
            break


if __name__ == "__main__":
    main()
