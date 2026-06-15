from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from pingpong_rl.envs.self_play_variety_env import SelfPlayVarietyConfig, SelfPlayVarietyEnv
from train.eval_utils import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the trained Stage 12 self-play policy.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage12")
    parser.add_argument("--opponent-model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage11")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = SelfPlayVarietyConfig(opponent_model_path=str(args.opponent_model_path))
    env = SelfPlayVarietyEnv(render_mode="human", config=config)
    model = PPO.load(args.model_path)
    obs, _ = env.reset(seed=args.seed)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
