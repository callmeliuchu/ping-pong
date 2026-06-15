from __future__ import annotations

import numpy as np
import pygame


class PygameRenderer:
    def __init__(self, width: int, height: int):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Ping Pong RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)

    def render_catch(self, env, mode: str = "human"):
        self._pump_events(env)
        self._draw_background()
        self._draw_ball(env.ball_x, env.ball_y, env.config.ball_radius)
        self._draw_paddle(
            env.config.agent_x,
            env.agent_y,
            env.config.paddle_width,
            env.config.paddle_height,
            (90, 170, 255),
        )
        return self._finish(env, mode)

    def render_pong(self, env, mode: str = "human"):
        self._pump_events(env)
        self._draw_background()
        self._draw_ball(env.ball_x, env.ball_y, env.config.ball_radius)
        self._draw_paddle(
            env.config.agent_x,
            env.agent_y,
            env.config.paddle_width,
            env.config.paddle_height,
            (90, 170, 255),
        )
        self._draw_paddle(
            env.opponent_x,
            env.opponent_y,
            env.config.paddle_width,
            env.config.paddle_height,
            (255, 120, 120),
        )
        return self._finish(env, mode)

    def render_gravity(self, env, mode: str = "human"):
        self._pump_events(env)
        self.screen.fill((18, 24, 30))
        pygame.draw.rect(
            self.screen,
            (58, 108, 124),
            pygame.Rect(
                env.config.table_left,
                env.config.table_y,
                env.config.table_right - env.config.table_left,
                12,
            ),
        )
        pygame.draw.line(
            self.screen,
            (220, 230, 235),
            (env.config.net_x, env.config.table_y),
            (env.config.net_x, env.config.table_y - env.config.net_height),
            3,
        )
        self._draw_trail(getattr(env, "ball_trail", []))
        self._draw_ball(env.ball_x, env.ball_y, env.config.ball_radius)
        if hasattr(env, "agent_angle"):
            self._draw_rotated_paddle(
                env.agent_x,
                env.agent_y,
                env.config.paddle_width,
                env.config.paddle_height,
                env.agent_angle,
                (90, 170, 255),
            )
            self._draw_rotated_paddle(
                env.opponent_x,
                env.opponent_y,
                env.config.paddle_width,
                env.config.paddle_height,
                env.opponent_angle,
                (255, 120, 120),
            )
        else:
            self._draw_paddle(
                env.config.paddle_x,
                env.agent_y,
                env.config.paddle_width,
                env.config.paddle_height,
                (90, 170, 255),
            )
            self._draw_paddle(
                env.config.opponent_x,
                env.opponent_y,
                env.config.paddle_width,
                env.config.paddle_height,
                (255, 120, 120),
            )
        if hasattr(env, "rally_length"):
            if hasattr(env, "style_match_landings"):
                gravity_mode = f"variety {getattr(env, 'last_target_style', 'none')}"
            elif hasattr(env, "compact_technique_hits"):
                gravity_mode = f"compact {getattr(env, 'last_stroke_type', 'none')}"
            elif hasattr(env, "loop_attempts"):
                gravity_mode = f"advanced {getattr(env, 'last_stroke_type', 'none')}"
            elif hasattr(env, "attack_attempts"):
                gravity_mode = "competitive"
            elif hasattr(env, "ball_spin"):
                gravity_mode = "realistic"
            else:
                gravity_mode = "rules" if getattr(env.config, "rules_enabled", False) else "rally"
            status = (
                f"{gravity_mode}  rally {env.rally_length}/{env.config.target_rally_length}  "
                f"spin {getattr(env, 'ball_spin', 0.0):.1f}  reason {getattr(env, 'point_reason', 'in_play')}"
            )
            text = self.font.render(status, True, (220, 230, 235))
            self.screen.blit(text, (16, 14))
        return self._finish(env, mode)

    def _pump_events(self, env) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.close()
                raise SystemExit

    def _draw_background(self) -> None:
        self.screen.fill((20, 24, 28))
        pygame.draw.line(
            self.screen,
            (70, 76, 84),
            (self.width // 2, 0),
            (self.width // 2, self.height),
            2,
        )

    def _draw_ball(self, x: float, y: float, radius: int) -> None:
        pygame.draw.circle(
            self.screen,
            (242, 244, 248),
            (int(x), int(y)),
            radius,
        )

    def _draw_trail(self, trail: list[tuple[float, float]]) -> None:
        if len(trail) < 2:
            return
        points = [(int(x), int(y)) for x, y in trail[-70:]]
        for index, point in enumerate(points):
            alpha = index / max(len(points) - 1, 1)
            color = (
                int(90 + 120 * alpha),
                int(150 + 80 * alpha),
                int(255 - 70 * alpha),
            )
            radius = 2 if index < len(points) - 10 else 3
            pygame.draw.circle(self.screen, color, point, radius)

    def _draw_paddle(self, x: float, y: float, width: int, height: int, color: tuple[int, int, int]) -> None:
        pygame.draw.rect(
            self.screen,
            color,
            pygame.Rect(
                x - width / 2,
                y - height / 2,
                width,
                height,
            ),
        )

    def _draw_rotated_paddle(
        self,
        x: float,
        y: float,
        width: int,
        height: int,
        angle: float,
        color: tuple[int, int, int],
    ) -> None:
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        half_w = width / 2
        half_h = height / 2
        corners = []
        for local_x, local_y in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)):
            world_x = x + cos_a * local_x + sin_a * local_y
            world_y = y - sin_a * local_x + cos_a * local_y
            corners.append((int(world_x), int(world_y)))
        pygame.draw.polygon(self.screen, color, corners)

    def _finish(self, env, mode: str):
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
