from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from pingpong_rl.envs import CompactTechniqueEnv
from train.eval_utils import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the trained Stage 10 compact-paddle policy.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage10")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = CompactTechniqueEnv(render_mode="human")
    model = PPO.load(args.model_path)
    obs, _ = env.reset(seed=args.seed)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
