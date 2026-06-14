from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pingpong_rl.envs import PongEnv, make_pong_config
from train.eval_utils import evaluate_pong, write_metrics


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5 bootstrap self-play from an existing PPO model.")
    parser.add_argument("--base-model-path", type=Path, default=ROOT / "models" / "ppo_pong_stage4")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_pong_stage5")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-json-path", type=Path, default=None)
    args = parser.parse_args()

    if not args.base_model_path.with_suffix(".zip").exists() and not args.base_model_path.exists():
        raise FileNotFoundError(f"Base model not found: {args.base_model_path}")

    def make_env():
        def _init():
            config = make_pong_config(5, str(args.base_model_path))
            return Monitor(PongEnv(render_mode=None, config=config))

        return _init

    vec_env = DummyVecEnv([make_env() for _ in range(args.num_envs)])
    vec_env.seed(args.seed)
    eval_env = Monitor(PongEnv(render_mode=None, config=make_pong_config(5, str(args.base_model_path))))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ROOT / "models" / "best" / "stage5"),
        log_path=str(ROOT / "logs" / "ppo_pong_stage5" / "eval"),
        eval_freq=max(args.timesteps // 10 // args.num_envs, 1),
        n_eval_episodes=min(args.eval_episodes, 50),
        deterministic=True,
        render=False,
    )
    model = PPO.load(args.base_model_path, env=vec_env)
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    vec_env.close()

    metrics = evaluate_pong(
        stage=5,
        model_path=args.model_path,
        opponent_model_path=str(args.base_model_path),
        episodes=args.eval_episodes,
        seed=args.seed,
    )
    write_metrics(metrics, args.eval_json_path or ROOT / "logs" / "ppo_pong_stage5" / "final_eval.json")


if __name__ == "__main__":
    main()
