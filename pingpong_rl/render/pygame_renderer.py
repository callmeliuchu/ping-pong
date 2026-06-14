from __future__ import annotations

import numpy as np
import pygame


class PygameRenderer:
    def __init__(self, width: int, height: int):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Ping Pong RL - Stage 1 Catch")
        self.clock = pygame.time.Clock()

    def render_catch(self, env, mode: str = "human"):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                raise SystemExit

        self.screen.fill((20, 24, 28))
        pygame.draw.line(
            self.screen,
            (70, 76, 84),
            (env.config.width // 2, 0),
            (env.config.width // 2, env.config.height),
            2,
        )
        pygame.draw.circle(
            self.screen,
            (242, 244, 248),
            (int(env.ball_x), int(env.ball_y)),
            env.config.ball_radius,
        )
        pygame.draw.rect(
            self.screen,
            (90, 170, 255),
            pygame.Rect(
                env.config.agent_x - env.config.paddle_width / 2,
                env.agent_y - env.config.paddle_height / 2,
                env.config.paddle_width,
                env.config.paddle_height,
            ),
        )

        if mode == "rgb_array":
            return np.transpose(
                pygame.surfarray.array3d(self.screen),
                axes=(1, 0, 2),
            )

        pygame.display.flip()
        self.clock.tick(env.metadata["render_fps"])
        return None

    def close(self):
        pygame.quit()

