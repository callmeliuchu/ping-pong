from __future__ import annotations

import pygame

from pingpong_rl.envs import CatchEnv


def main() -> None:
    env = CatchEnv(render_mode="human")
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

