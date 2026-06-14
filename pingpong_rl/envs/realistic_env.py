from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class RealisticPingPongConfig:
    width: int = 900
    height: int = 500
    table_y: float = 330.0
    table_left: float = 120.0
    table_right: float = 780.0
    net_x: float = 450.0
    net_height: float = 54.0
    ball_radius: int = 7
    gravity: float = 0.22
    spin_lift: float = 0.012
    spin_bounce_coupling: float = 0.07
    bounce_damping: float = 0.86
    agent_x_min: float = 130.0
    agent_x_max: float = 250.0
    opponent_x_min: float = 650.0
    opponent_x_max: float = 770.0
    paddle_width: int = 16
    paddle_height: int = 120
    paddle_x_speed: float = 6.0
    paddle_y_speed: float = 11.0
    paddle_angle_speed: float = 0.12
    max_paddle_angle: float = 0.75
    opponent_speed: float = 10.0
    opponent_x_speed: float = 4.0
    max_ball_speed: float = 15.0
    max_spin: float = 9.0
    ball_speed_x_min: float = 5.2
    ball_speed_x_max: float = 6.8
    max_steps: int = 3200
    target_rally_length: int = 10
    assist_residual_scale: float = 0.35
    hit_reward: float = 0.9
    legal_landing_reward: float = 0.35
    spin_reward: float = 0.03
    angle_reward: float = 0.02
    rally_success_reward: float = 2.5
    score_reward: float = 1.5
    miss_penalty: float = -1.0
    shaping_scale: float = 0.06
    move_penalty: float = 0.001


class RealisticPingPongEnv(gym.Env):
    """Stage 7: gravity ping pong with movable, rotatable paddles and spin."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, config: RealisticPingPongConfig | None = None):
        super().__init__()
        self.config = config or RealisticPingPongConfig()
        self.render_mode = render_mode
        self.renderer = None
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)

        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.ball_spin = 0.0
        self.agent_x = 0.0
        self.agent_y = 0.0
        self.agent_vx = 0.0
        self.agent_vy = 0.0
        self.agent_angle = 0.0
        self.opponent_x = 0.0
        self.opponent_y = 0.0
        self.opponent_vx = 0.0
        self.opponent_vy = 0.0
        self.opponent_angle = 0.0
        self.steps = 0
        self.rally_length = 0
        self.agent_hits = 0
        self.opponent_hits = 0
        self.legal_landings = 0
        self.last_hitter = "opponent"
        self.required_landing_side = "opponent"
        self.shot_landed = False
        self.side_bounces = 0
        self.serve_phase = True
        self.serve_first_landing_done = False
        self.point_winner = None
        self.point_reason = "in_play"
        self.last_contact_quality = 0.0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.ball_x = float(self.config.opponent_x_max - 18)
        self.ball_y = float(self.np_random.uniform(self.config.table_y - 112, self.config.table_y - 82))
        self.ball_vx = -float(self.np_random.uniform(self.config.ball_speed_x_min, self.config.ball_speed_x_max))
        self.agent_x = self.config.agent_x_min + 55
        self.agent_y = self.config.table_y - 95
        self.agent_vx = 0.0
        self.agent_vy = 0.0
        self.agent_angle = 0.0
        self.opponent_x = self.config.opponent_x_max - 55
        self.opponent_y = self.config.table_y - 95
        self.opponent_vx = 0.0
        self.opponent_vy = 0.0
        self.opponent_angle = 0.0
        self.ball_vy = self._sample_legal_serve_velocity()
        self.ball_spin = float(self.np_random.uniform(-0.6, 0.6))
        self.steps = 0
        self.rally_length = 0
        self.agent_hits = 0
        self.opponent_hits = 0
        self.legal_landings = 0
        self.last_hitter = "opponent"
        self.required_landing_side = "opponent"
        self.shot_landed = False
        self.side_bounces = 0
        self.serve_phase = True
        self.serve_first_landing_done = False
        self.point_winner = None
        self.point_reason = "in_play"
        self.last_contact_quality = 0.0
        return self._get_obs(), self._get_info(False, False, False, False, False)

    def step(self, action):
        self.steps += 1
        action_arr = np.asarray(action, dtype=np.float32)
        previous_distance = self._distance_to_agent_target()
        self._apply_agent_action(action_arr)
        self._apply_opponent_policy()
        self._move_ball()

        agent_hit = self._paddle_collision(self.agent_x, self.agent_y, self.agent_angle) and self.ball_vx < 0
        opponent_hit = self._paddle_collision(self.opponent_x, self.opponent_y, self.opponent_angle) and self.ball_vx > 0
        legal_landing = False
        if agent_hit:
            if self._can_hit("agent"):
                self.rally_length += 1
                self.agent_hits += 1
                self._bounce_from_paddle("agent")
                self._mark_hit("agent")
            else:
                agent_hit = False
        elif opponent_hit:
            if self._can_hit("opponent"):
                self.rally_length += 1
                self.opponent_hits += 1
                self._bounce_from_paddle("opponent")
                self._mark_hit("opponent")
            else:
                opponent_hit = False

        table_side, net_hit = self._table_bounce()
        if self.point_winner is None:
            legal_landing = self._apply_rules_after_bounce(table_side, net_hit)
            self._apply_rules_after_out()

        agent_score = self.point_winner == "agent"
        agent_miss = self.point_winner == "opponent"
        rally_success = self.legal_landings >= self.config.target_rally_length + 2
        reward = self._reward(previous_distance, action_arr, agent_hit, legal_landing, agent_score, agent_miss, rally_success)
        terminated = bool(agent_score or agent_miss or rally_success)
        truncated = bool(self.steps >= self.config.max_steps and not terminated)
        info = self._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return None
        if self.renderer is None:
            from pingpong_rl.render.pygame_renderer import PygameRenderer

            self.renderer = PygameRenderer(self.config.width, self.config.height)
        return self.renderer.render_gravity(self, mode=self.render_mode)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def _apply_agent_action(self, action: np.ndarray) -> None:
        if self.ball_vx < 0:
            target_y = self.predict_ball_y_at_x(self.agent_x)
            target_x = self.config.agent_x_min + 55
            desired_angle = np.clip((target_y - self.agent_y) / 150.0, -1.0, 1.0) * self.config.max_paddle_angle
        else:
            target_y = self.config.table_y - 95
            target_x = self.config.agent_x_min + 55
            desired_angle = 0.0
        base_vx = float(np.clip(target_x - self.agent_x, -self.config.paddle_x_speed, self.config.paddle_x_speed))
        base_vy = float(np.clip(target_y - self.agent_y, -self.config.paddle_y_speed, self.config.paddle_y_speed))
        base_angle_delta = float(np.clip(desired_angle - self.agent_angle, -self.config.paddle_angle_speed, self.config.paddle_angle_speed))
        residual = self.config.assist_residual_scale
        self.agent_vx = float(base_vx + np.clip(action[0], -1.0, 1.0) * self.config.paddle_x_speed * residual)
        self.agent_vy = float(base_vy + np.clip(action[1], -1.0, 1.0) * self.config.paddle_y_speed * residual)
        self.agent_x = float(np.clip(self.agent_x + self.agent_vx, self.config.agent_x_min, self.config.agent_x_max))
        self.agent_y = self._clamp_paddle_y(self.agent_y + self.agent_vy)
        self.agent_angle = float(
            np.clip(
                self.agent_angle + base_angle_delta + np.clip(action[2], -1.0, 1.0) * self.config.paddle_angle_speed * residual,
                -self.config.max_paddle_angle,
                self.config.max_paddle_angle,
            )
        )

    def _apply_opponent_policy(self) -> None:
        if self.ball_vx > 0:
            target_y = self.predict_ball_y_at_x(self.opponent_x)
            target_x = self.config.opponent_x_max - 55
            desired_angle = -0.18 if self.ball_y < self.opponent_y else 0.18
        else:
            target_y = self.config.table_y - 95
            target_x = self.config.opponent_x_max - 55
            desired_angle = 0.0
        self.opponent_vy = float(np.clip(target_y - self.opponent_y, -self.config.opponent_speed, self.config.opponent_speed))
        self.opponent_vx = float(np.clip(target_x - self.opponent_x, -self.config.opponent_x_speed, self.config.opponent_x_speed))
        self.opponent_x = float(np.clip(self.opponent_x + self.opponent_vx, self.config.opponent_x_min, self.config.opponent_x_max))
        self.opponent_y = self._clamp_paddle_y(self.opponent_y + self.opponent_vy)
        self.opponent_angle = float(
            np.clip(
                self.opponent_angle + np.clip(desired_angle - self.opponent_angle, -0.08, 0.08),
                -self.config.max_paddle_angle,
                self.config.max_paddle_angle,
            )
        )

    def _move_ball(self) -> None:
        self.ball_vy += self.config.gravity - self.config.spin_lift * self.ball_spin
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        speed = float(np.hypot(self.ball_vx, self.ball_vy))
        if speed > self.config.max_ball_speed:
            scale = self.config.max_ball_speed / speed
            self.ball_vx *= scale
            self.ball_vy *= scale

    def _mark_hit(self, player: str) -> None:
        self.last_hitter = player
        self.required_landing_side = "opponent" if player == "agent" else "agent"
        self.shot_landed = False
        self.side_bounces = 0
        self.serve_phase = False
        self.point_reason = "in_play"

    def _can_hit(self, player: str) -> bool:
        return bool(self.required_landing_side == player and self.shot_landed)

    def _award_point(self, winner: str, reason: str) -> None:
        if self.point_winner is None:
            self.point_winner = winner
            self.point_reason = reason

    def _apply_rules_after_bounce(self, table_side: str | None, net_hit: bool) -> bool:
        if net_hit:
            self._award_point("opponent" if self.last_hitter == "agent" else "agent", "net_fault")
            return False
        if table_side is None:
            return False
        if self.serve_phase and not self.serve_first_landing_done:
            if table_side != "opponent":
                self._award_point("agent", "serve_first_bounce_fault")
                return False
            self.serve_first_landing_done = True
            self.required_landing_side = "agent"
            self.shot_landed = False
            self.side_bounces = 0
            self.legal_landings += 1
            self.ball_vy -= 0.35
            return True
        if table_side != self.required_landing_side:
            self._award_point("opponent" if self.last_hitter == "agent" else "agent", "wrong_side_landing")
            return False
        if not self.shot_landed:
            self.shot_landed = True
            self.side_bounces = 1
            self.legal_landings += 1
            return True
        self.side_bounces += 1
        if self.side_bounces >= 2:
            self._award_point("agent" if self.last_hitter == "agent" else "opponent", "second_bounce")
        return False

    def _apply_rules_after_out(self) -> None:
        if self.point_winner is not None:
            return
        out_left = self.ball_x < -self.config.ball_radius
        out_right = self.ball_x > self.config.width + self.config.ball_radius
        fell_low = self.ball_y > self.config.height + self.config.ball_radius
        if not (out_left or out_right or fell_low):
            return
        if self.shot_landed:
            self._award_point("agent" if self.last_hitter == "agent" else "opponent", "receiver_missed")
        else:
            self._award_point("opponent" if self.last_hitter == "agent" else "agent", "out_before_landing")

    def _table_bounce(self) -> tuple[str | None, bool]:
        table_side = None
        over_table = self.config.table_left <= self.ball_x <= self.config.table_right
        above_table = self.ball_y + self.config.ball_radius >= self.config.table_y and self.ball_vy > 0
        if over_table and above_table:
            self.ball_y = self.config.table_y - self.config.ball_radius
            self.ball_vy *= -self.config.bounce_damping
            self.ball_vx += self.ball_spin * self.config.spin_bounce_coupling
            self.ball_vx *= 0.995
            self.ball_spin *= 0.72
            table_side = "agent" if self.ball_x < self.config.net_x else "opponent"

        net_top = self.config.table_y - self.config.net_height
        net_hit = bool(abs(self.ball_x - self.config.net_x) <= self.config.ball_radius and self.ball_y + self.config.ball_radius >= net_top)
        return table_side, net_hit

    def _paddle_collision(self, paddle_x: float, paddle_y: float, angle: float) -> bool:
        dx = self.ball_x - paddle_x
        dy = self.ball_y - paddle_y
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        local_x = cos_a * dx - sin_a * dy
        local_y = sin_a * dx + cos_a * dy
        return bool(
            abs(local_x) <= self.config.paddle_width / 2 + self.config.ball_radius
            and abs(local_y) <= self.config.paddle_height / 2 + self.config.ball_radius
        )

    def _bounce_from_paddle(self, player: str) -> None:
        if player == "agent":
            direction = 1
            paddle_x = self.agent_x
            paddle_y = self.agent_y
            paddle_vx = self.agent_vx
            paddle_vy = self.agent_vy
            angle = self.agent_angle
            landing_min = self.config.net_x + 70
            landing_max = self.opponent_x - 70
            base_landing_x = (landing_min + landing_max) / 2
        else:
            direction = -1
            paddle_x = self.opponent_x
            paddle_y = self.opponent_y
            paddle_vx = self.opponent_vx
            paddle_vy = self.opponent_vy
            angle = self.opponent_angle
            landing_min = self.agent_x + 120
            landing_max = self.config.net_x - 55
            base_landing_x = (landing_min + landing_max) / 2

        offset = float(np.clip((self.ball_y - paddle_y) / (self.config.paddle_height / 2), -1.0, 1.0))
        self.last_contact_quality = float(1.0 - min(abs(offset), 1.0))
        self.ball_x = paddle_x + direction * (self.config.paddle_width / 2 + self.config.ball_radius + 1)
        speed = min(max(abs(self.ball_vx) * 0.98 + 0.25 + abs(paddle_vx) * 0.06, self.config.ball_speed_x_min), 8.2)
        self.ball_vx = direction * speed
        landing_x = base_landing_x + direction * angle * 95.0 + offset * 38.0 + paddle_vx * 2.2
        landing_x = float(np.clip(landing_x, landing_min, landing_max))
        self.ball_vy = self._aimed_vertical_velocity(landing_x, self.config.table_y - self.config.ball_radius)
        self.ball_vy -= 0.75 + max(0.0, abs(angle) - 0.15) * 0.55
        spin_delta = angle * 4.8 + paddle_vy * 0.08 - offset * 2.0
        self.ball_spin = float(np.clip(self.ball_spin * 0.45 + spin_delta, -self.config.max_spin, self.config.max_spin))

    def predict_ball_y_at_x(self, target_x: float) -> float:
        if abs(self.ball_vx) < 1e-6 or (target_x - self.ball_x) * self.ball_vx <= 0:
            return self.ball_y
        x = self.ball_x
        y = self.ball_y
        vx = self.ball_vx
        vy = self.ball_vy
        spin = self.ball_spin
        previous_x = x
        previous_y = y
        for _ in range(700):
            previous_x = x
            previous_y = y
            vy += self.config.gravity - self.config.spin_lift * spin
            x += vx
            y += vy
            if (target_x - previous_x) * (target_x - x) <= 0:
                span = x - previous_x
                alpha = 0.0 if abs(span) < 1e-6 else (target_x - previous_x) / span
                predicted_y = previous_y + alpha * (y - previous_y)
                return float(np.clip(predicted_y, self.config.table_y - 205, self.config.table_y - self.config.paddle_height / 2))
            over_table = self.config.table_left <= x <= self.config.table_right
            above_table = y + self.config.ball_radius >= self.config.table_y and vy > 0
            if over_table and above_table:
                y = self.config.table_y - self.config.ball_radius
                vy *= -self.config.bounce_damping
                vx += spin * self.config.spin_bounce_coupling
                vx *= 0.995
                spin *= 0.72
            net_top = self.config.table_y - self.config.net_height
            if abs(x - self.config.net_x) <= self.config.ball_radius and y + self.config.ball_radius >= net_top:
                break
        return float(np.clip(y, self.config.table_y - 205, self.config.table_y - self.config.paddle_height / 2))

    def _distance_to_agent_target(self) -> float:
        target_y = self.predict_ball_y_at_x(self.agent_x) if self.ball_vx < 0 else self.config.table_y - 95
        target_x = self.agent_x if self.ball_vx < 0 else self.config.agent_x_min + 55
        return float(abs(target_y - self.agent_y) + 0.25 * abs(target_x - self.agent_x))

    def _aimed_vertical_velocity(self, target_x: float, target_y: float) -> float:
        if abs(self.ball_vx) < 1e-6:
            return 0.0
        t = abs((target_x - self.ball_x) / self.ball_vx)
        if t <= 1e-6:
            return 0.0
        vy = (target_y - self.ball_y - 0.5 * self.config.gravity * t * t) / t
        return float(np.clip(vy, -self.config.max_ball_speed, self.config.max_ball_speed))

    def _sample_legal_serve_velocity(self) -> float:
        candidates = []
        for vy in np.linspace(-7.0, 9.0, 129):
            second_bounce_x = self._simulate_serve_second_bounce_x(float(vy))
            if second_bounce_x is None:
                continue
            if self.agent_x + 110 <= second_bounce_x <= self.config.net_x - 65:
                candidates.append(float(vy))
        if candidates:
            return float(self.np_random.choice(candidates))
        landing_x = float(self.np_random.uniform(self.config.net_x + 80, self.config.net_x + 150))
        return self._aimed_vertical_velocity(landing_x, self.config.table_y - self.config.ball_radius)

    def _simulate_serve_second_bounce_x(self, initial_vy: float) -> float | None:
        x = self.ball_x
        y = self.ball_y
        vx = self.ball_vx
        vy = initial_vy
        bounce_sides = []
        for _ in range(500):
            vy += self.config.gravity
            x += vx
            y += vy
            net_top = self.config.table_y - self.config.net_height
            if abs(x - self.config.net_x) <= self.config.ball_radius and y + self.config.ball_radius >= net_top:
                return None
            over_table = self.config.table_left <= x <= self.config.table_right
            above_table = y + self.config.ball_radius >= self.config.table_y and vy > 0
            if over_table and above_table:
                side = "agent" if x < self.config.net_x else "opponent"
                bounce_sides.append(side)
                if len(bounce_sides) == 1:
                    if side != "opponent":
                        return None
                    y = self.config.table_y - self.config.ball_radius
                    vy *= -self.config.bounce_damping
                    vy -= 0.35
                    vx *= 0.995
                elif len(bounce_sides) == 2:
                    return float(x) if side == "agent" else None
            if x < -self.config.ball_radius or y > self.config.height + self.config.ball_radius:
                return None
        return None

    def _reward(
        self,
        previous_distance: float,
        action: np.ndarray,
        agent_hit: bool,
        legal_landing: bool,
        agent_score: bool,
        agent_miss: bool,
        rally_success: bool,
    ) -> float:
        reward = 0.0
        if self.ball_vx < 0:
            reward += self.config.shaping_scale * (previous_distance - self._distance_to_agent_target()) / self.config.height
            target_y = self.predict_ball_y_at_x(self.agent_x)
            reward -= 0.0008 * abs(target_y - self.agent_y)
        reward -= self.config.move_penalty * float(np.sum(np.square(action)))
        if agent_hit:
            reward += self.config.hit_reward
            reward += self.config.long_rally_reward if hasattr(self.config, "long_rally_reward") else 0.0
            reward += self.config.angle_reward * min(abs(self.agent_angle), self.config.max_paddle_angle)
            reward += self.config.spin_reward * min(abs(self.ball_spin), self.config.max_spin)
            reward += 0.2 * self.last_contact_quality
        if legal_landing and self.last_hitter == "agent":
            reward += self.config.legal_landing_reward
        if agent_score:
            reward += self.config.score_reward
        if rally_success:
            reward += self.config.rally_success_reward
        elif agent_miss:
            reward += self.config.miss_penalty
        return float(reward)

    def _get_obs(self) -> np.ndarray:
        target_y = self.predict_ball_y_at_x(self.agent_x) if self.ball_vx < 0 else self.config.table_y - 95
        desired_angle = np.clip((target_y - self.agent_y) / 150.0, -1.0, 1.0) * self.config.max_paddle_angle
        obs = np.array(
            [
                self.ball_x / self.config.width * 2 - 1,
                self.ball_y / self.config.height * 2 - 1,
                self.ball_vx / self.config.max_ball_speed,
                self.ball_vy / self.config.max_ball_speed,
                self.ball_spin / self.config.max_spin,
                self.agent_x / self.config.width * 2 - 1,
                self.agent_y / self.config.height * 2 - 1,
                self.agent_vx / self.config.paddle_x_speed,
                self.agent_vy / self.config.paddle_y_speed,
                self.agent_angle / self.config.max_paddle_angle,
                self.opponent_x / self.config.width * 2 - 1,
                self.opponent_y / self.config.height * 2 - 1,
                self.opponent_angle / self.config.max_paddle_angle,
                target_y / self.config.height * 2 - 1,
                (target_y - self.agent_y) / self.config.height,
                desired_angle / self.config.max_paddle_angle,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _get_info(
        self,
        agent_hit: bool,
        agent_score: bool,
        agent_miss: bool,
        rally_success: bool,
        legal_landing: bool,
    ) -> dict:
        return {
            "agent_hit": agent_hit,
            "agent_score": agent_score,
            "agent_miss": agent_miss,
            "rally_success": rally_success,
            "legal_landing": legal_landing,
            "rules_enabled": True,
            "point_winner": self.point_winner,
            "point_reason": self.point_reason,
            "last_hitter": self.last_hitter,
            "shot_landed": self.shot_landed,
            "serve_phase": self.serve_phase,
            "serve_first_landing_done": self.serve_first_landing_done,
            "legal_landings": self.legal_landings,
            "agent_hits": self.agent_hits,
            "opponent_hits": self.opponent_hits,
            "rally_length": self.rally_length,
            "avg_abs_spin": abs(self.ball_spin),
            "agent_angle": self.agent_angle,
            "contact_quality": self.last_contact_quality,
            "steps": self.steps,
        }

    def _clamp_paddle_y(self, y: float) -> float:
        min_y = self.config.table_y - 210
        max_y = self.config.table_y - self.config.paddle_height / 2
        return float(np.clip(y, min_y, max_y))
