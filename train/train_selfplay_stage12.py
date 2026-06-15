from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pingpong_rl.envs.self_play_variety_env import SelfPlayVarietyConfig, SelfPlayVarietyEnv
from train.eval_utils import ROOT, write_metrics
from train.evaluate_selfplay import evaluate_selfplay_model


def make_env(opponent_model_path: Path):
    def _init():
        config = SelfPlayVarietyConfig(opponent_model_path=str(opponent_model_path))
        return Monitor(SelfPlayVarietyEnv(render_mode=None, config=config))

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 12 league-style self-play from a historical model.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--base-model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage11")
    parser.add_argument("--opponent-model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage11")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_selfplay_stage12")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-json-path", type=Path, default=None)
    args = parser.parse_args()

    check_env(SelfPlayVarietyEnv(config=SelfPlayVarietyConfig(opponent_model_path=str(args.opponent_model_path))))

    league_dir = ROOT / "models" / "selfplay" / "stage12"
    league_dir.mkdir(parents=True, exist_ok=True)
    opponent_snapshot = league_dir / f"gen_{args.generation - 1}.zip"
    if not opponent_snapshot.exists():
        shutil.copy2(args.opponent_model_path.with_suffix(".zip"), opponent_snapshot)

    vec_env = DummyVecEnv([make_env(args.opponent_model_path) for _ in range(args.num_envs)])
    vec_env.seed(args.seed)
    eval_env = Monitor(SelfPlayVarietyEnv(config=SelfPlayVarietyConfig(opponent_model_path=str(args.opponent_model_path))))

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.timesteps // 4 // args.num_envs, 1),
        save_path=str(ROOT / "models" / "checkpoints"),
        name_prefix=f"ppo_selfplay_stage12_gen{args.generation}",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ROOT / "models" / "best" / "stage12"),
        log_path=str(ROOT / "logs" / "ppo_selfplay_stage12" / "eval"),
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

    metrics = evaluate_selfplay_model(args.model_path, args.opponent_model_path, args.eval_episodes, seed=args.seed)
    write_metrics(metrics, args.eval_json_path or ROOT / "logs" / "ppo_selfplay_stage12" / f"gen_{args.generation}_eval.json")


if __name__ == "__main__":
    main()
