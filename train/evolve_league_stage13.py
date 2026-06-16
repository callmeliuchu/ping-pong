from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pingpong_rl.envs.league_self_play_env import LeagueSelfPlayConfig, LeagueSelfPlayEnv, RedLeagueSelfPlayEnv
from train.eval_utils import ROOT, write_metrics
from train.evaluate_league_stage13 import default_opponent_paths, evaluate_league_model, evaluate_red_challenge
from train.train_league_stage13 import make_env


def _zip_path(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _unique_existing(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    seen_hashes: set[str] = set()
    for path in paths:
        source = _zip_path(path)
        if not source.exists():
            continue
        resolved = source.resolve()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if resolved in seen or digest in seen_hashes:
            continue
        unique.append(path)
        seen.add(resolved)
        seen_hashes.add(digest)
    return unique


def league_pool(champion_path: Path, include_generations: bool = True) -> list[Path]:
    candidates = default_opponent_paths() + [champion_path]
    if include_generations:
        candidates.extend(sorted((ROOT / "models" / "selfplay" / "stage13").glob("gen_*.zip")))
    return _unique_existing(candidates)


def _score(metrics: dict[str, Any]) -> float:
    return (
        100.0 * float(metrics["pool_win_rate"])
        + 35.0 * float(metrics["worst_opponent_win_rate"])
        + 4.0 * float(metrics["avg_rally_length"])
        + 0.04 * float(metrics["estimated_elo_delta_vs_pool"])
    )


def _should_promote(
    candidate: dict[str, Any],
    champion: dict[str, Any],
    red_challenge: dict[str, Any] | None,
    min_score_improvement: float,
    min_pool_win_rate: float,
    min_worst_win_rate: float,
    min_rally_length: float,
    min_red_challenge_win_rate: float,
) -> tuple[bool, str]:
    candidate_score = _score(candidate)
    champion_score = _score(champion)
    if red_challenge is not None:
        red_win_rate = float(red_challenge["red_win_rate"])
        red_normal_end_rate = float(red_challenge["normal_end_rate"])
        red_rally_length = float(red_challenge["avg_rally_length"])
        if (
            red_win_rate >= min_red_challenge_win_rate
            and red_normal_end_rate >= 0.95
            and red_rally_length >= min_rally_length
            and float(candidate["pool_win_rate"]) >= min_pool_win_rate
        ):
            return (
                True,
                "red challenger win_rate "
                f"{red_win_rate:.3f} >= {min_red_challenge_win_rate:.3f} "
                f"against current blue champion",
            )
    if float(candidate["pool_win_rate"]) < min_pool_win_rate:
        return False, f"pool_win_rate {candidate['pool_win_rate']:.3f} < {min_pool_win_rate:.3f}"
    if float(candidate["worst_opponent_win_rate"]) < min_worst_win_rate:
        return False, f"worst_opponent_win_rate {candidate['worst_opponent_win_rate']:.3f} < {min_worst_win_rate:.3f}"
    if float(candidate["avg_rally_length"]) < min_rally_length:
        return False, f"avg_rally_length {candidate['avg_rally_length']:.2f} < {min_rally_length:.2f}"
    if candidate_score < champion_score + min_score_improvement:
        return False, f"score {candidate_score:.2f} < champion score {champion_score:.2f} + {min_score_improvement:.2f}"
    return True, f"score {candidate_score:.2f} >= champion score {champion_score:.2f} + {min_score_improvement:.2f}"


def _should_promote_red_champion(
    red_challenge: dict[str, Any],
    min_red_challenge_win_rate: float,
    min_rally_length: float,
) -> tuple[bool, str]:
    red_win_rate = float(red_challenge["red_win_rate"])
    normal_end_rate = float(red_challenge["normal_end_rate"])
    avg_rally_length = float(red_challenge["avg_rally_length"])
    if red_win_rate < min_red_challenge_win_rate:
        return False, f"red_win_rate {red_win_rate:.3f} < {min_red_challenge_win_rate:.3f}"
    if normal_end_rate < 0.85:
        return False, f"red normal_end_rate {normal_end_rate:.3f} < 0.850"
    if avg_rally_length < min_rally_length:
        return False, f"red avg_rally_length {avg_rally_length:.2f} < {min_rally_length:.2f}"
    return True, (
        f"red challenger win_rate {red_win_rate:.3f} >= {min_red_challenge_win_rate:.3f}, "
        f"normal_end_rate {normal_end_rate:.3f}, avg_rally_length {avg_rally_length:.2f}"
    )


def _train_candidate(
    base_model_path: Path,
    opponent_paths: list[Path],
    candidate_path: Path,
    generation: int,
    timesteps: int,
    num_envs: int,
    seed: int,
    eval_episodes: int,
    train_side: str,
) -> Path:
    if train_side == "red":
        def make_red_env():
            config = LeagueSelfPlayConfig(opponent_model_paths=tuple(str(path) for path in opponent_paths))
            return Monitor(RedLeagueSelfPlayEnv(render_mode=None, config=config))

        env_fns = [make_red_env for _ in range(num_envs)]
        eval_config = LeagueSelfPlayConfig(opponent_model_paths=tuple(str(path) for path in opponent_paths))
        eval_env = Monitor(RedLeagueSelfPlayEnv(config=eval_config))
    else:
        env_fns = [make_env(opponent_paths) for _ in range(num_envs)]
        eval_config = LeagueSelfPlayConfig(opponent_model_paths=tuple(str(path) for path in opponent_paths))
        eval_env = Monitor(LeagueSelfPlayEnv(config=eval_config))

    vec_env = DummyVecEnv(env_fns)
    vec_env.seed(seed)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(timesteps // 4 // num_envs, 1),
        save_path=str(ROOT / "models" / "checkpoints"),
        name_prefix=f"ppo_evolve_stage13_{train_side}_gen{generation}",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ROOT / "models" / "best" / f"stage13_{train_side}_gen{generation}"),
        log_path=str(ROOT / "logs" / "ppo_league_stage13" / f"evolve_{train_side}_gen{generation}"),
        eval_freq=max(timesteps // 8 // num_envs, 1),
        n_eval_episodes=min(eval_episodes, 50),
        deterministic=True,
        render=False,
    )
    model = PPO.load(base_model_path, env=vec_env)
    model.verbose = 1
    best_model_path = ROOT / "models" / "best" / f"stage13_{train_side}_gen{generation}" / "best_model"
    model.learn(total_timesteps=timesteps, callback=CallbackList([checkpoint_callback, eval_callback]))
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(candidate_path)
    vec_env.close()
    eval_env.close()
    return best_model_path if _zip_path(best_model_path).exists() else candidate_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously evolve Stage 13 and promote only stronger candidates.")
    parser.add_argument("--start-generation", type=int, default=3)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--timesteps-per-generation", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument("--champion-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage13")
    parser.add_argument("--red-champion-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage13_red")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--min-score-improvement", type=float, default=0.50)
    parser.add_argument("--min-pool-win-rate", type=float, default=0.49)
    parser.add_argument("--min-worst-win-rate", type=float, default=0.45)
    parser.add_argument("--min-rally-length", type=float, default=3.8)
    parser.add_argument("--min-red-challenge-win-rate", type=float, default=0.54)
    parser.add_argument("--train-side", choices=("blue", "red", "alternate"), default="blue")
    parser.add_argument("--history-path", type=Path, default=ROOT / "logs" / "ppo_league_stage13" / "evolution_history.json")
    args = parser.parse_args()

    league_dir = ROOT / "models" / "selfplay" / "stage13"
    league_dir.mkdir(parents=True, exist_ok=True)
    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if args.history_path.exists():
        history = json.loads(args.history_path.read_text())

    champion_path = args.champion_path
    for offset in range(args.generations):
        generation = args.start_generation + offset
        if args.train_side == "alternate":
            train_side = "red" if generation % 2 == 0 else "blue"
        else:
            train_side = args.train_side
        opponent_paths = league_pool(champion_path)
        champion_metrics = evaluate_league_model(champion_path, opponent_paths, args.eval_episodes, seed=args.seed + offset * 1000)
        champion_json = ROOT / "logs" / "ppo_league_stage13" / f"champion_before_gen{generation}.json"
        write_metrics(champion_metrics, champion_json)

        candidate_path = league_dir / f"gen_{generation}"
        best_candidate_path = _train_candidate(
            champion_path,
            opponent_paths,
            candidate_path,
            generation,
            args.timesteps_per_generation,
            args.num_envs,
            args.seed + generation,
            args.eval_episodes,
            train_side,
        )
        candidate_metrics = evaluate_league_model(candidate_path, opponent_paths, args.eval_episodes, seed=args.seed + offset * 1000)
        candidate_json = ROOT / "logs" / "ppo_league_stage13" / f"gen{generation}_evolve_eval.json"
        write_metrics(candidate_metrics, candidate_json)
        best_candidate_metrics = None
        if _zip_path(best_candidate_path).resolve() != _zip_path(candidate_path).resolve():
            best_candidate_metrics = evaluate_league_model(
                best_candidate_path,
                opponent_paths,
                args.eval_episodes,
                seed=args.seed + offset * 1000,
            )
            best_candidate_json = ROOT / "logs" / "ppo_league_stage13" / f"gen{generation}_best_evolve_eval.json"
            write_metrics(best_candidate_metrics, best_candidate_json)
            if _score(best_candidate_metrics) > _score(candidate_metrics):
                shutil.copy2(_zip_path(best_candidate_path), _zip_path(candidate_path))
                candidate_metrics = best_candidate_metrics

        red_challenge_metrics = evaluate_red_challenge(
            candidate_path,
            champion_path,
            args.eval_episodes,
            seed=args.seed + offset * 1000 + 500_000,
        )
        red_challenge_json = ROOT / "logs" / "ppo_league_stage13" / f"gen{generation}_red_challenge_eval.json"
        write_metrics(red_challenge_metrics, red_challenge_json)

        promoted, reason = _should_promote(
            candidate_metrics,
            champion_metrics,
            red_challenge_metrics,
            args.min_score_improvement,
            args.min_pool_win_rate,
            args.min_worst_win_rate,
            args.min_rally_length,
            args.min_red_challenge_win_rate,
        )
        if promoted:
            shutil.copy2(_zip_path(candidate_path), _zip_path(args.champion_path))
            champion_path = args.champion_path
        red_promoted, red_reason = _should_promote_red_champion(
            red_challenge_metrics,
            args.min_red_challenge_win_rate,
            args.min_rally_length,
        )
        if red_promoted:
            args.red_champion_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_zip_path(candidate_path), _zip_path(args.red_champion_path))

        record = {
            "generation": generation,
            "train_side": train_side,
            "candidate_path": str(_zip_path(candidate_path)),
            "best_checkpoint_path": str(_zip_path(best_candidate_path)),
            "opponent_pool": [str(path) for path in opponent_paths],
            "promoted": promoted,
            "reason": reason,
            "red_promoted": red_promoted,
            "red_reason": red_reason,
            "champion_score": _score(champion_metrics),
            "candidate_score": _score(candidate_metrics),
            "champion_metrics": champion_metrics,
            "candidate_metrics": candidate_metrics,
            "red_challenge_metrics": red_challenge_metrics,
            "final_candidate_metrics": None if best_candidate_metrics is None else json.loads(candidate_json.read_text()),
        }
        history.append(record)
        args.history_path.write_text(json.dumps(history, indent=2, sort_keys=True))
        print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
