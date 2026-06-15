from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pingpong_rl.envs.league_self_play_env import LeagueSelfPlayConfig, LeagueSelfPlayEnv
from train.eval_utils import ROOT, write_metrics
from train.evaluate_league_stage13 import default_opponent_paths, evaluate_league_model


def _zip_source(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _unique_existing(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        source = _zip_source(path)
        if not source.exists():
            continue
        resolved = source.resolve()
        if resolved in seen:
            continue
        unique.append(path)
        seen.add(resolved)
    return unique


def make_env(opponent_model_paths: list[Path]):
    def _init():
        config = LeagueSelfPlayConfig(opponent_model_paths=tuple(str(path) for path in opponent_model_paths))
        return Monitor(LeagueSelfPlayEnv(render_mode=None, config=config))

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 13 with a multi-model league opponent pool.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation", type=int, default=2)
    parser.add_argument("--base-model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage12")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_league_stage13")
    parser.add_argument("--opponent-model-paths", type=Path, nargs="*", default=None)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-json-path", type=Path, default=None)
    args = parser.parse_args()

    opponent_paths = _unique_existing(args.opponent_model_paths or default_opponent_paths())
    if not opponent_paths:
        raise ValueError("Stage 13 training needs at least one opponent model.")

    config = LeagueSelfPlayConfig(opponent_model_paths=tuple(str(path) for path in opponent_paths))
    check_env(LeagueSelfPlayEnv(config=config))

    league_dir = ROOT / "models" / "selfplay" / "stage13"
    league_dir.mkdir(parents=True, exist_ok=True)
    for index, opponent_path in enumerate(opponent_paths):
        snapshot_path = league_dir / f"pool_{index}_{opponent_path.with_suffix('').name}.zip"
        if not snapshot_path.exists():
            shutil.copy2(_zip_source(opponent_path), snapshot_path)

    vec_env = DummyVecEnv([make_env(opponent_paths) for _ in range(args.num_envs)])
    vec_env.seed(args.seed)
    eval_env = Monitor(LeagueSelfPlayEnv(config=config))

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.timesteps // 4 // args.num_envs, 1),
        save_path=str(ROOT / "models" / "checkpoints"),
        name_prefix=f"ppo_league_stage13_gen{args.generation}",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ROOT / "models" / "best" / "stage13"),
        log_path=str(ROOT / "logs" / "ppo_league_stage13" / "eval"),
        eval_freq=max(args.timesteps // 8 // args.num_envs, 1),
        n_eval_episodes=min(args.eval_episodes, 50),
        deterministic=True,
        render=False,
    )

    model = PPO.load(args.base_model_path, env=vec_env)
    model.verbose = 1
    model.learn(total_timesteps=args.timesteps, callback=CallbackList([checkpoint_callback, eval_callback]))
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    generation_path = league_dir / f"gen_{args.generation}.zip"
    model.save(generation_path)
    vec_env.close()

    metrics = evaluate_league_model(args.model_path, opponent_paths, args.eval_episodes, seed=args.seed)
    write_metrics(metrics, args.eval_json_path or ROOT / "logs" / "ppo_league_stage13" / f"gen_{args.generation}_eval.json")


if __name__ == "__main__":
    main()
