from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from pingpong_rl.envs import RobotArmPingPongEnv
from train.eval_utils import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the Stage 14 joint-controlled robot-arm policy.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_robot_arm_stage14")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = RobotArmPingPongEnv(render_mode="human")
    model = PPO.load(args.model_path)
    obs, _ = env.reset(seed=args.seed)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
