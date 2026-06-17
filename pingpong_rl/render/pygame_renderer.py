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
        if hasattr(env, "agent_angle"):
            if hasattr(env, "agent_joint_angles"):
                self._draw_articulated_robot_arm(
                    "left",
                    getattr(env, "agent_arm_base"),
                    env.agent_joint_angles,
                    env.config.arm_upper_length,
                    env.config.arm_forearm_length,
                    env.config.arm_hand_length,
                    (90, 170, 255),
                )
                self._draw_articulated_robot_arm(
                    "right",
                    getattr(env, "opponent_arm_base"),
                    env.opponent_joint_angles,
                    env.config.arm_upper_length,
                    env.config.arm_forearm_length,
                    env.config.arm_hand_length,
                    (255, 120, 120),
                )
            else:
                self._draw_robot_arm(
                    "left",
                    env.agent_x,
                    env.agent_y,
                    env.agent_angle,
                    (90, 170, 255),
                    env.config.table_left,
                    env.config.table_y,
                )
                self._draw_robot_arm(
                    "right",
                    env.opponent_x,
                    env.opponent_y,
                    env.opponent_angle,
                    (255, 120, 120),
                    env.config.table_right,
                    env.config.table_y,
                )
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
        self._draw_ball(env.ball_x, env.ball_y, env.config.ball_radius)
        if hasattr(env, "rally_length"):
            if hasattr(env, "style_match_landings"):
                gravity_mode = f"variety {getattr(env, 'last_target_style', 'none')}"
            elif hasattr(env, "compact_technique_hits"):
                gravity_mode = f"compact {getattr(env, 'last_stroke_type', 'none')}"
            elif getattr(env, "agent_joint_angles", None) is not None:
                gravity_mode = f"robot_arm {getattr(env, 'last_stroke_type', 'none')}"
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

    def _draw_robot_arm(
        self,
        side: str,
        paddle_x: float,
        paddle_y: float,
        paddle_angle: float,
        color: tuple[int, int, int],
        table_edge_x: float,
        table_y: float,
    ) -> None:
        direction = 1 if side == "left" else -1
        base_x = table_edge_x - direction * 62.0
        base_y = table_y + 76.0
        shoulder_x = base_x + direction * 18.0
        shoulder_y = base_y - 46.0
        wrist_x = paddle_x - direction * 18.0 * float(np.cos(paddle_angle))
        wrist_y = paddle_y + 18.0 * float(np.sin(paddle_angle))
        elbow_x, elbow_y = self._two_link_elbow(shoulder_x, shoulder_y, wrist_x, wrist_y, side)

        base_color = (72, 82, 92)
        joint_color = (222, 230, 238)
        shadow_color = (10, 14, 18)
        arm_color = tuple(int(0.62 * component + 0.38 * 190) for component in color)

        pygame.draw.rect(
            self.screen,
            base_color,
            pygame.Rect(base_x - 18, base_y - 8, 36, 18),
            border_radius=4,
        )
        pygame.draw.line(self.screen, shadow_color, (shoulder_x, shoulder_y), (elbow_x, elbow_y), 14)
        pygame.draw.line(self.screen, shadow_color, (elbow_x, elbow_y), (wrist_x, wrist_y), 12)
        pygame.draw.line(self.screen, arm_color, (shoulder_x, shoulder_y), (elbow_x, elbow_y), 9)
        pygame.draw.line(self.screen, arm_color, (elbow_x, elbow_y), (wrist_x, wrist_y), 8)
        pygame.draw.line(
            self.screen,
            joint_color,
            (wrist_x, wrist_y),
            (paddle_x, paddle_y),
            5,
        )
        for joint_x, joint_y, radius in (
            (shoulder_x, shoulder_y, 11),
            (elbow_x, elbow_y, 10),
            (wrist_x, wrist_y, 8),
        ):
            pygame.draw.circle(self.screen, shadow_color, (int(joint_x), int(joint_y)), radius + 3)
            pygame.draw.circle(self.screen, joint_color, (int(joint_x), int(joint_y)), radius)
            pygame.draw.circle(self.screen, arm_color, (int(joint_x), int(joint_y)), max(radius - 5, 3))

    def _two_link_elbow(
        self,
        shoulder_x: float,
        shoulder_y: float,
        wrist_x: float,
        wrist_y: float,
        side: str,
    ) -> tuple[int, int]:
        upper = 134.0
        forearm = 128.0
        dx = wrist_x - shoulder_x
        dy = wrist_y - shoulder_y
        distance = float(np.hypot(dx, dy))
        distance = max(1.0, min(distance, upper + forearm - 1.0))
        base_angle = float(np.arctan2(dy, dx))
        cos_offset = (upper * upper + distance * distance - forearm * forearm) / (2.0 * upper * distance)
        offset = float(np.arccos(np.clip(cos_offset, -1.0, 1.0)))
        bend = 1.0 if side == "left" else -1.0
        elbow_angle = base_angle + bend * offset
        elbow_x = shoulder_x + upper * float(np.cos(elbow_angle))
        elbow_y = shoulder_y + upper * float(np.sin(elbow_angle))
        return int(elbow_x), int(elbow_y)

    def _draw_articulated_robot_arm(
        self,
        side: str,
        base: tuple[float, float],
        joint_angles: np.ndarray,
        upper: float,
        forearm: float,
        hand: float,
        color: tuple[int, int, int],
    ) -> None:
        direction = 1.0 if side == "left" else -1.0
        base_x, base_y = base
        shoulder_x = base_x + direction * 18.0
        shoulder_y = base_y - 46.0
        base_heading = 0.0 if side == "left" else float(np.pi)
        theta0 = base_heading + float(joint_angles[0])
        theta1 = theta0 + float(joint_angles[1])
        paddle_angle = float(joint_angles[2])
        elbow_x = shoulder_x + upper * float(np.cos(theta0))
        elbow_y = shoulder_y + upper * float(np.sin(theta0))
        wrist_x = elbow_x + forearm * float(np.cos(theta1))
        wrist_y = elbow_y + forearm * float(np.sin(theta1))
        paddle_x = wrist_x + direction * hand * float(np.cos(paddle_angle))
        paddle_y = wrist_y - hand * float(np.sin(paddle_angle))

        base_color = (72, 82, 92)
        joint_color = (222, 230, 238)
        shadow_color = (10, 14, 18)
        arm_color = tuple(int(0.62 * component + 0.38 * 190) for component in color)
        pygame.draw.rect(
            self.screen,
            base_color,
            pygame.Rect(base_x - 18, base_y - 8, 36, 18),
            border_radius=4,
        )
        pygame.draw.line(self.screen, shadow_color, (shoulder_x, shoulder_y), (elbow_x, elbow_y), 14)
        pygame.draw.line(self.screen, shadow_color, (elbow_x, elbow_y), (wrist_x, wrist_y), 12)
        pygame.draw.line(self.screen, arm_color, (shoulder_x, shoulder_y), (elbow_x, elbow_y), 9)
        pygame.draw.line(self.screen, arm_color, (elbow_x, elbow_y), (wrist_x, wrist_y), 8)
        pygame.draw.line(self.screen, joint_color, (wrist_x, wrist_y), (paddle_x, paddle_y), 5)
        for joint_x, joint_y, radius in (
            (shoulder_x, shoulder_y, 11),
            (elbow_x, elbow_y, 10),
            (wrist_x, wrist_y, 8),
        ):
            pygame.draw.circle(self.screen, shadow_color, (int(joint_x), int(joint_y)), radius + 3)
            pygame.draw.circle(self.screen, joint_color, (int(joint_x), int(joint_y)), radius)
            pygame.draw.circle(self.screen, arm_color, (int(joint_x), int(joint_y)), max(radius - 5, 3))

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
