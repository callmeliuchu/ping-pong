from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from pingpong_rl.envs.league_self_play_env import LeagueSelfPlayConfig, LeagueSelfPlayEnv
from train.eval_utils import ROOT
from train.evaluate_league_stage13 import default_opponent_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the trained Stage 13 league self-play policy.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage13")
    parser.add_argument("--opponent-model-paths", type=Path, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    opponent_paths = args.opponent_model_paths if args.opponent_model_paths is not None else default_opponent_paths()
    config = LeagueSelfPlayConfig(opponent_model_paths=tuple(str(path) for path in opponent_paths))
    env = LeagueSelfPlayEnv(render_mode="human", config=config)
    model = PPO.load(args.model_path)
    obs, _ = env.reset(seed=args.seed)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
