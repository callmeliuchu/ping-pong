from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from stable_baselines3 import PPO

from pingpong_rl.envs.robot_arm_red_scoring_bilateral_env import (
    RobotArmRedScoringBilateralConfig,
    RobotArmRedScoringBilateralEnv,
)
from train.eval_utils import ROOT
from train.evaluate_robot_arm_red_scoring_bilateral import default_red_scoring_bilateral_opponent_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the Stage 23 red-scoring bilateral robot-arm league.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage23_blue.zip")
    parser.add_argument("--opponent-model-paths", type=Path, nargs="*", default=None)
    parser.add_argument("--red-model-path", type=Path, default=None)
    parser.add_argument("--profile-index", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gif-path", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=1600)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    if args.red_model_path is not None:
        opponent_paths = [args.red_model_path]
    else:
        opponent_paths = args.opponent_model_paths if args.opponent_model_paths is not None else default_red_scoring_bilateral_opponent_paths()
    config = RobotArmRedScoringBilateralConfig(
        opponent_model_paths=tuple(str(path) for path in opponent_paths),
        pressure_profile_index=args.profile_index,
    )
    model = PPO.load(args.model_path)

    if args.gif_path is not None:
        env = RobotArmRedScoringBilateralEnv(render_mode="rgb_array", config=config)
        obs, _ = env.reset(seed=args.seed)
        frames = []
        for step in range(args.max_steps):
            frame = env.render()
            if frame is not None and step % 2 == 0:
                frames.append(Image.fromarray(frame).convert("P", palette=Image.Palette.ADAPTIVE))
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        env.close()
        if not frames:
            raise RuntimeError("No frames were rendered.")
        args.gif_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            args.gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=max(1, int(1000 / args.fps)),
            loop=0,
            optimize=False,
        )
        print(args.gif_path)
        return

    env = RobotArmRedScoringBilateralEnv(render_mode="human", config=config)
    obs, _ = env.reset(seed=args.seed)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
