from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class GravityPingPongConfig:
    width: int = 900
    height: int = 500
    table_y: float = 330.0
    table_left: float = 120.0
    table_right: float = 780.0
    net_x: float = 450.0
    net_height: float = 54.0
    ball_radius: int = 7
    gravity: float = 0.22
    bounce_damping: float = 0.86
    ball_speed_x_min: float = 5.0
    ball_speed_x_max: float = 7.0
    ball_speed_y_min: float = -6.0
    ball_speed_y_max: float = -2.0
    paddle_x: float = 170.0
    opponent_x: float = 730.0
    paddle_width: int = 14
    paddle_height: int = 140
    paddle_speed: float = 10.0
    opponent_speed: float = 10.0
    max_ball_speed: float = 14.0
    max_steps: int = 3000
    max_table_bounces: int = 10
    target_rally_length: int = 20
    rules_enabled: bool = True
    hit_reward: float = 0.8
    legal_landing_reward: float = 0.25
    net_clearance_lift: float = 1.2
    serve_net_clearance_lift: float = 0.45
    opponent_return_reward: float = 0.15
    long_rally_reward: float = 0.08
    rally_success_reward: float = 2.0
    score_reward: float = 1.5
    miss_penalty: float = -1.0
    shaping_scale: float = 0.04
    alignment_reward: float = 0.004
    move_penalty: float = 0.001


class GravityPingPongEnv(gym.Env):
    """Stage 6 prototype: side-view table tennis with gravity, table and net."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, config: GravityPingPongConfig | None = None):
        super().__init__()
        self.config = config or GravityPingPongConfig()
        self.render_mode = render_mode
        self.renderer = None
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)

        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.agent_y = 0.0
        self.agent_vy = 0.0
        self.opponent_y = 0.0
        self.opponent_vy = 0.0
        self.steps = 0
        self.rally_length = 0
        self.agent_touched = False
        self.table_bounces = 0
        self.agent_hits = 0
        self.opponent_hits = 0
        self.last_hitter = "opponent"
        self.required_landing_side = "agent"
        self.shot_landed = False
        self.side_bounces = 0
        self.serve_phase = True
        self.serve_first_landing_done = False
        self.legal_landings = 0
        self.point_winner = None
        self.point_reason = "in_play"

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.ball_x = float(self.config.opponent_x - 16)
        self.ball_y = float(self.np_random.uniform(self.config.table_y - 115, self.config.table_y - 80))
        self.ball_vx = -float(self.np_random.uniform(self.config.ball_speed_x_min, self.config.ball_speed_x_max))
        self.ball_vy = self._sample_legal_serve_velocity()
        self.agent_y = self.config.table_y - 90
        self.opponent_y = self.config.table_y - 90
        self.agent_vy = 0.0
        self.opponent_vy = 0.0
        self.steps = 0
        self.rally_length = 0
        self.agent_touched = False
        self.table_bounces = 0
        self.agent_hits = 0
        self.opponent_hits = 0
        self.last_hitter = "opponent"
        self.required_landing_side = "opponent"
        self.shot_landed = False
        self.side_bounces = 0
        self.serve_phase = True
        self.serve_first_landing_done = False
        self.legal_landings = 0
        self.point_winner = None
        self.point_reason = "in_play"
        return self._get_obs(), self._get_info(False, False, False, False, False)

    def step(self, action: int):
        self.steps += 1
        previous_distance = self._distance_to_agent_target()
        self._apply_agent_action(int(action))
        self._apply_opponent_policy()

        self.ball_vy += self.config.gravity
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        agent_hit = self._paddle_collision(self.config.paddle_x, self.agent_y) and self.ball_vx < 0
        opponent_hit = self._paddle_collision(self.config.opponent_x, self.opponent_y) and self.ball_vx > 0
        legal_landing = False
        if agent_hit:
            if self.config.rules_enabled and not self._can_hit("agent"):
                agent_hit = False
            else:
                self.agent_touched = True
                self.rally_length += 1
                self.agent_hits += 1
                self.table_bounces = 0
                self.ball_x = self.config.paddle_x + self.config.paddle_width / 2 + self.config.ball_radius
                self._bounce_from_paddle(self.agent_y, self.agent_vy, direction=1)
                self._mark_hit("agent")
        elif opponent_hit:
            if self.config.rules_enabled and not self._can_hit("opponent"):
                opponent_hit = False
            else:
                self.rally_length += 1
                self.opponent_hits += 1
                self.table_bounces = 0
                self.ball_x = self.config.opponent_x - self.config.paddle_width / 2 - self.config.ball_radius
                self._bounce_from_paddle(self.opponent_y, self.opponent_vy, direction=-1)
                self._mark_hit("opponent")

        table_side, net_hit = self._table_bounce()
        if self.config.rules_enabled and self.point_winner is None:
            legal_landing = self._apply_rules_after_bounce(table_side, net_hit)
            self._apply_rules_after_out()

        if self.config.rules_enabled:
            agent_score = self.point_winner == "agent"
            agent_miss = self.point_winner == "opponent"
        else:
            agent_score = self._agent_scores()
            agent_miss = self._agent_misses()
        if self.config.rules_enabled:
            rally_success = self.legal_landings >= self.config.target_rally_length + 2
        else:
            rally_success = self.rally_length >= self.config.target_rally_length
        reward = self._reward(
            previous_distance,
            int(action),
            agent_hit,
            opponent_hit,
            legal_landing,
            agent_score,
            agent_miss,
            rally_success,
        )
        terminated = bool(agent_score or agent_miss or rally_success)
        truncated = bool(self.steps >= self.config.max_steps and not terminated)
        info = self._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def _mark_hit(self, player: str) -> None:
        if not self.config.rules_enabled:
            return
        self.last_hitter = player
        self.required_landing_side = "opponent" if player == "agent" else "agent"
        self.shot_landed = False
        self.side_bounces = 0
        self.serve_phase = False
        self.point_reason = "in_play"

    def _can_hit(self, player: str) -> bool:
        if not self.config.rules_enabled:
            return True
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
            self.ball_vy -= self.config.serve_net_clearance_lift
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

    def _apply_agent_action(self, action: int) -> None:
        self.agent_vy = self._action_to_velocity(action, self.config.paddle_speed)
        self.agent_y = self._clamp_paddle(self.agent_y + self.agent_vy)

    def _apply_opponent_policy(self) -> None:
        if self.ball_vx > 0:
            target_y = self.predict_ball_y_at_x(self.config.opponent_x)
        else:
            target_y = self.config.table_y - 90
        if target_y < self.opponent_y - 5:
            self.opponent_vy = -self.config.opponent_speed
        elif target_y > self.opponent_y + 5:
            self.opponent_vy = self.config.opponent_speed
        else:
            self.opponent_vy = 0.0
        self.opponent_y = self._clamp_paddle(self.opponent_y + self.opponent_vy)

    def predict_ball_y_at_x(self, target_x: float) -> float:
        if abs(self.ball_vx) < 1e-6:
            return self.ball_y
        if (target_x - self.ball_x) * self.ball_vx <= 0:
            return self.ball_y
        x = self.ball_x
        y = self.ball_y
        vx = self.ball_vx
        vy = self.ball_vy
        previous_x = x
        previous_y = y
        for _ in range(600):
            previous_x = x
            previous_y = y
            vy += self.config.gravity
            x += vx
            y += vy
            if (target_x - previous_x) * (target_x - x) <= 0:
                span = x - previous_x
                alpha = 0.0 if abs(span) < 1e-6 else (target_x - previous_x) / span
                predicted_y = previous_y + alpha * (y - previous_y)
                return float(np.clip(predicted_y, self.config.table_y - 190, self.config.table_y - self.config.paddle_height / 2))
            over_table = self.config.table_left <= x <= self.config.table_right
            above_table = y + self.config.ball_radius >= self.config.table_y and vy > 0
            if over_table and above_table:
                y = self.config.table_y - self.config.ball_radius
                vy *= -self.config.bounce_damping
                vx *= 0.995
            net_top = self.config.table_y - self.config.net_height
            if abs(x - self.config.net_x) <= self.config.ball_radius and y + self.config.ball_radius >= net_top:
                break
        return float(np.clip(y, self.config.table_y - 190, self.config.table_y - self.config.paddle_height / 2))

    def _distance_to_agent_target(self) -> float:
        if self.ball_vx < 0:
            target_y = self.predict_ball_y_at_x(self.config.paddle_x)
        else:
            target_y = self.config.table_y - 90
        return abs(target_y - self.agent_y)

    def _table_bounce(self) -> tuple[str | None, bool]:
        table_side = None
        over_table = self.config.table_left <= self.ball_x <= self.config.table_right
        above_table = self.ball_y + self.config.ball_radius >= self.config.table_y and self.ball_vy > 0
        if over_table and above_table:
            self.ball_y = self.config.table_y - self.config.ball_radius
            self.ball_vy *= -self.config.bounce_damping
            self.ball_vx *= 0.995
            self.table_bounces += 1
            table_side = "agent" if self.ball_x < self.config.net_x else "opponent"

        net_top = self.config.table_y - self.config.net_height
        crosses_net = abs(self.ball_x - self.config.net_x) <= self.config.ball_radius
        net_hit = False
        if crosses_net and self.ball_y + self.config.ball_radius >= net_top:
            net_hit = True
            if not self.config.rules_enabled:
                self.ball_vx *= -0.6
                self.ball_x = self.config.net_x - np.sign(self.ball_vx) * (self.config.ball_radius + 1)
        return table_side, net_hit

    def _agent_scores(self) -> bool:
        fell_low = self.ball_y > self.config.height + self.config.ball_radius
        dead_on_opponent_side = self.table_bounces >= self.config.max_table_bounces and self.ball_x > self.config.net_x
        return bool(self.agent_touched and (fell_low and self.ball_x > self.config.net_x or dead_on_opponent_side))

    def _agent_misses(self) -> bool:
        left_out = self.ball_x < -self.config.ball_radius
        fell_low = self.ball_y > self.config.height + self.config.ball_radius
        fell_on_agent_side = fell_low and self.ball_x <= self.config.net_x
        never_returned = fell_low and not self.agent_touched
        dead_on_agent_side = self.table_bounces >= self.config.max_table_bounces and self.ball_x <= self.config.net_x
        dead_without_touch = self.table_bounces >= self.config.max_table_bounces and not self.agent_touched
        return bool(left_out or fell_on_agent_side or never_returned or dead_on_agent_side or dead_without_touch)

    def _paddle_collision(self, paddle_x: float, paddle_y: float) -> bool:
        return bool(
            abs(self.ball_x - paddle_x) <= self.config.ball_radius + self.config.paddle_width / 2
            and abs(self.ball_y - paddle_y) <= self.config.ball_radius + self.config.paddle_height / 2
        )

    def _bounce_from_paddle(self, paddle_y: float, paddle_vy: float, direction: int) -> None:
        self.ball_vx = direction * min(max(abs(self.ball_vx) * 0.99 + 0.05, self.config.ball_speed_x_min), 8.0)
        offset = float(np.clip((self.ball_y - paddle_y) / (self.config.paddle_height / 2), -1.0, 1.0))
        if self.config.rules_enabled:
            if direction > 0:
                base_landing_x = (self.config.net_x + self.config.opponent_x) / 2
                landing_x = base_landing_x + offset * 45.0 + paddle_vy * 1.8
                landing_x = float(np.clip(landing_x, self.config.net_x + 70, self.config.opponent_x - 80))
            else:
                base_landing_x = (self.config.paddle_x + self.config.net_x) / 2 + 35
                landing_x = base_landing_x + offset * 45.0 + paddle_vy * 1.8
                landing_x = float(np.clip(landing_x, self.config.paddle_x + 150, self.config.net_x - 50))
            self.ball_vy = self._aimed_vertical_velocity(
                landing_x,
                self.config.table_y - self.config.ball_radius,
            ) - self.config.net_clearance_lift
        else:
            target_x = self.config.opponent_x if direction > 0 else self.config.paddle_x
            target_y = self.config.table_y - 95 + offset * 28.0 + paddle_vy * 1.4
            target_y = float(np.clip(target_y, self.config.table_y - 150, self.config.table_y - 65))
            self.ball_vy = self._aimed_vertical_velocity(target_x, target_y)

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
            if self.config.paddle_x + 110 <= second_bounce_x <= self.config.net_x - 70:
                candidates.append(float(vy))
        if not candidates:
            landing_x = float(self.np_random.uniform(self.config.net_x + 80, self.config.net_x + 150))
            return self._aimed_vertical_velocity(landing_x, self.config.table_y - self.config.ball_radius)
        return float(self.np_random.choice(candidates))

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
                    vy -= self.config.serve_net_clearance_lift
                    vx *= 0.995
                elif len(bounce_sides) == 2:
                    return float(x) if side == "agent" else None
            if x < -self.config.ball_radius or y > self.config.height + self.config.ball_radius:
                return None
        return None

    def _reward(
        self,
        previous_distance: float,
        action: int,
        agent_hit: bool,
        opponent_hit: bool,
        legal_landing: bool,
        agent_score: bool,
        agent_miss: bool,
        rally_success: bool,
    ) -> float:
        reward = 0.0
        if self.ball_vx < 0:
            reward += self.config.shaping_scale * (previous_distance - self._distance_to_agent_target()) / self.config.height
            target_y = self.predict_ball_y_at_x(self.config.paddle_x)
            if target_y < self.agent_y - 5:
                reward += self.config.alignment_reward if action == 1 else -self.config.alignment_reward
            elif target_y > self.agent_y + 5:
                reward += self.config.alignment_reward if action == 2 else -self.config.alignment_reward
        if action != 0:
            reward -= self.config.move_penalty
        if agent_hit:
            reward += self.config.hit_reward
            reward += self.config.long_rally_reward * min(self.rally_length, 20)
        if opponent_hit:
            reward += self.config.opponent_return_reward
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
        target_y = self.predict_ball_y_at_x(self.config.paddle_x) if self.ball_vx < 0 else self.config.table_y - 90
        obs = np.array(
            [
                self.ball_x / self.config.width * 2 - 1,
                self.ball_y / self.config.height * 2 - 1,
                self.ball_vx / self.config.max_ball_speed,
                self.ball_vy / self.config.max_ball_speed,
                self.agent_y / self.config.height * 2 - 1,
                self.agent_vy / self.config.paddle_speed,
                self.opponent_y / self.config.height * 2 - 1,
                self.opponent_vy / self.config.opponent_speed,
                target_y / self.config.height * 2 - 1,
                (target_y - self.agent_y) / self.config.height,
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
            "rules_enabled": self.config.rules_enabled,
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
            "table_bounces": self.table_bounces,
            "steps": self.steps,
        }

    def _clamp_paddle(self, y: float) -> float:
        min_y = self.config.table_y - 190
        max_y = self.config.table_y - self.config.paddle_height / 2
        return float(np.clip(y, min_y, max_y))

    @staticmethod
    def _action_to_velocity(action: int, speed: float) -> float:
        if action == 1:
            return -speed
        if action == 2:
            return speed
        return 0.0
