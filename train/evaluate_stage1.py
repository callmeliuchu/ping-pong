from __future__ import annotations

import argparse
from pathlib import Path

from train.eval_utils import ROOT, evaluate_stage1, print_metrics, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Stage 1 catch policy.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_catch_stage1")
    parser.add_argument("--episodes", type=int, default=100, help="Use 0 with --render to watch until the window is closed.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.render and args.episodes == 0:
        from stable_baselines3 import PPO
        from pingpong_rl.envs import CatchEnv

        env = CatchEnv(render_mode="human")
        model = PPO.load(args.model_path)
        obs, _ = env.reset(seed=args.seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                obs, _ = env.reset()

    metrics = evaluate_stage1(args.model_path, args.episodes, args.render, args.seed)
    write_metrics(metrics, args.json_path)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
