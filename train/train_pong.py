from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pingpong_rl.envs import PongEnv, make_pong_config
from train.eval_utils import evaluate_pong, write_metrics


ROOT = Path(__file__).resolve().parents[1]


def make_env(stage: int, opponent_model_path: str | None = None):
    def _init():
        return Monitor(PongEnv(render_mode=None, config=make_pong_config(stage, opponent_model_path)))

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pong stages 2-5 with PPO.")
    parser.add_argument("--stage", type=int, choices=[2, 3, 4, 5], required=True)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opponent-model-path", type=str, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--load-model-path", type=Path, default=None)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--eval-json-path", type=Path, default=None)
    parser.add_argument("--best-model-dir", type=Path, default=None)
    parser.add_argument("--no-eval-callback", action="store_true")
    args = parser.parse_args()

    default_timesteps = {2: 500_000, 3: 1_000_000, 4: 1_500_000, 5: 1_000_000}
    timesteps = args.timesteps or default_timesteps[args.stage]
    model_path = args.model_path or ROOT / "models" / f"ppo_pong_stage{args.stage}"

    config = make_pong_config(args.stage, args.opponent_model_path)
    check_env(PongEnv(config=config))

    vec_env = DummyVecEnv([make_env(args.stage, args.opponent_model_path) for _ in range(args.num_envs)])
    vec_env.seed(args.seed)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(timesteps // 5 // args.num_envs, 1),
        save_path=str(ROOT / "models" / "checkpoints"),
        name_prefix=f"ppo_pong_stage{args.stage}",
    )
    callbacks = [checkpoint_callback]

    if not args.no_eval_callback:
        eval_env = Monitor(PongEnv(render_mode=None, config=config))
        best_model_dir = args.best_model_dir or ROOT / "models" / "best" / f"stage{args.stage}"
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(ROOT / "logs" / f"ppo_pong_stage{args.stage}" / "eval"),
                eval_freq=args.eval_freq or max(timesteps // 10 // args.num_envs, 1),
                n_eval_episodes=min(args.eval_episodes, 50),
                deterministic=True,
                render=False,
            )
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
            ent_coef=0.01,
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(ROOT / "logs" / f"ppo_pong_stage{args.stage}"),
        )
    model.learn(total_timesteps=timesteps, callback=CallbackList(callbacks))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    vec_env.close()

    metrics = evaluate_pong(
        stage=args.stage,
        model_path=model_path,
        opponent_model_path=args.opponent_model_path,
        episodes=args.eval_episodes,
        seed=args.seed,
    )
    write_metrics(metrics, args.eval_json_path or ROOT / "logs" / f"ppo_pong_stage{args.stage}" / "final_eval.json")


if __name__ == "__main__":
    main()
