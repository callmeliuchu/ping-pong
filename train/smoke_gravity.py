from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3.common.env_checker import check_env

from pingpong_rl.envs import GravityPingPongEnv
from train.eval_utils import print_metrics, smoke_gravity, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the Stage 6 gravity environment.")
    parser.add_argument("--episodes", type=int, default=5, help="Use 0 with --render to watch until the window is closed.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    check_env(GravityPingPongEnv())
    if args.render and args.episodes == 0:
        env = GravityPingPongEnv(render_mode="human")
        obs, _ = env.reset(seed=args.seed)
        while True:
            action = 0
            if env.ball_vx < 0:
                target_y = env.predict_ball_y_at_x(env.config.paddle_x)
            else:
                target_y = env.config.table_y - 90
            if target_y < env.agent_y - 5:
                action = 1
            elif target_y > env.agent_y + 5:
                action = 2
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

    metrics = smoke_gravity(args.episodes, args.render, args.seed)
    write_metrics(metrics, args.json_path)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
