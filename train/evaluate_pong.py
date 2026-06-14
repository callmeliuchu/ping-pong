from __future__ import annotations

import argparse
from pathlib import Path

from train.eval_utils import ROOT, evaluate_pong, print_metrics, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pong stages 2-5.")
    parser.add_argument("--stage", type=int, choices=[2, 3, 4, 5], required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--opponent-model-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100, help="Use 0 with --render to watch until the window is closed.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.render and args.episodes == 0:
        from stable_baselines3 import PPO
        from pingpong_rl.envs import PongEnv, make_pong_config

        env = PongEnv(render_mode="human", config=make_pong_config(args.stage, args.opponent_model_path))
        model = PPO.load(args.model_path or ROOT / "models" / f"ppo_pong_stage{args.stage}")
        obs, _ = env.reset(seed=args.seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                obs, _ = env.reset()

    metrics = evaluate_pong(
        stage=args.stage,
        model_path=args.model_path or ROOT / "models" / f"ppo_pong_stage{args.stage}",
        opponent_model_path=args.opponent_model_path,
        episodes=args.episodes,
        render=args.render,
        seed=args.seed,
    )
    write_metrics(metrics, args.json_path)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
