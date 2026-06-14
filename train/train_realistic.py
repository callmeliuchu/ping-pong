from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pingpong_rl.envs import RealisticPingPongEnv
from train.eval_utils import ROOT, write_metrics
from train.evaluate_realistic import evaluate_realistic_model


def make_env():
    def _init():
        return Monitor(RealisticPingPongEnv(render_mode=None))

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 7 realistic paddle/spin ping pong with PPO.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_realistic_stage7")
    parser.add_argument("--load-model-path", type=Path, default=None)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-json-path", type=Path, default=None)
    args = parser.parse_args()

    check_env(RealisticPingPongEnv())

    vec_env = DummyVecEnv([make_env() for _ in range(args.num_envs)])
    vec_env.seed(args.seed)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.timesteps // 5 // args.num_envs, 1),
        save_path=str(ROOT / "models" / "checkpoints"),
        name_prefix="ppo_realistic_stage7",
    )
    eval_env = Monitor(RealisticPingPongEnv(render_mode=None))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ROOT / "models" / "best" / "stage7"),
        log_path=str(ROOT / "logs" / "ppo_realistic_stage7" / "eval"),
        eval_freq=max(args.timesteps // 10 // args.num_envs, 1),
        n_eval_episodes=min(args.eval_episodes, 50),
        deterministic=True,
        render=False,
    )

    if args.load_model_path is not None:
        model = PPO.load(args.load_model_path, env=vec_env)
        model.verbose = 1
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.015,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(ROOT / "logs" / "ppo_realistic_stage7"),
        )

    model.learn(total_timesteps=args.timesteps, callback=CallbackList([checkpoint_callback, eval_callback]))
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    vec_env.close()

    metrics = evaluate_realistic_model(args.model_path, args.eval_episodes, seed=args.seed)
    write_metrics(metrics, args.eval_json_path or ROOT / "logs" / "ppo_realistic_stage7" / "final_eval.json")


if __name__ == "__main__":
    main()
