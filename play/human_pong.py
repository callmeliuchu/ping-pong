from __future__ import annotations

import argparse

import pygame

from pingpong_rl.envs import PongEnv, make_pong_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Pong stages manually.")
    parser.add_argument("--stage", type=int, choices=[2, 3, 4], default=2)
    args = parser.parse_args()

    env = PongEnv(render_mode="human", config=make_pong_config(args.stage))
    env.reset()

    while True:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action = 2
        else:
            action = 0
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()


if __name__ == "__main__":
    main()
