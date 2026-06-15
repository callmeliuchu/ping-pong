from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from pingpong_rl.envs.realistic_env import RealisticPingPongConfig, RealisticPingPongEnv


@dataclass(frozen=True)
class CompetitivePingPongConfig(RealisticPingPongConfig):
    paddle_height: int = 145
    opponent_speed: float = 16.0
    opponent_x_speed: float = 7.0
    target_rally_length: int = 30
    assist_residual_scale: float = 0.65
    max_steps: int = 2600
    score_reward: float = 4.0
    miss_penalty: float = -3.0
    hit_reward: float = 0.45
    legal_landing_reward: float = 0.22
    attack_landing_reward: float = 0.45
    attack_score_bonus: float = 2.0
    deep_landing_x: float = 620.0
    opponent_base_error: float = 0.004
    opponent_attack_error_scale: float = 0.055
    randomize_opponent_level: bool = True


class CompetitiveRealisticEnv(RealisticPingPongEnv):
    """Stage 8: competitive point-play with realistic paddle motion and spin."""

    def __init__(self, render_mode: str | None = None, config: CompetitivePingPongConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or CompetitivePingPongConfig())
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(18,), dtype=np.float32)
        self.attack_intent = 0.0
        self.attack_attempts = 0
        self.attack_landings = 0
        self.attack_scores = 0
        self.agent_min_x_seen = self.agent_x
        self.agent_max_x_seen = self.agent_x
        self.last_agent_attack = False
        self.last_agent_deep_landing = False
        self.opponent_level = 3

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        self.attack_intent = 0.0
        self.attack_attempts = 0
        self.attack_landings = 0
        self.attack_scores = 0
        self.agent_min_x_seen = self.agent_x
        self.agent_max_x_seen = self.agent_x
        self.last_agent_attack = False
        self.last_agent_deep_landing = False
        if self.config.randomize_opponent_level:
            self.opponent_level = int(self.np_random.integers(1, 5))
        else:
            self.opponent_level = 3
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
                self.last_agent_attack = self.attack_intent > 0.25 or abs(self.agent_angle) > 0.34
                self.attack_attempts += int(self.last_agent_attack)
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
            if legal_landing and self.last_hitter == "agent":
                self.last_agent_deep_landing = self.ball_x >= self.config.deep_landing_x
                self.attack_landings += int(self.last_agent_attack or self.last_agent_deep_landing)
            self._apply_rules_after_out()
            self._maybe_force_opponent_error(legal_landing)

        agent_score = self.point_winner == "agent"
        agent_miss = self.point_winner == "opponent"
        if agent_score and (self.last_agent_attack or self.last_agent_deep_landing):
            self.attack_scores += 1
        rally_success = False
        reward = self._reward(previous_distance, action_arr, agent_hit, legal_landing, agent_score, agent_miss, rally_success)
        terminated = bool(agent_score or agent_miss)
        truncated = bool(self.steps >= self.config.max_steps and not terminated)
        info = self._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def _apply_agent_action(self, action: np.ndarray) -> None:
        self.attack_intent = float(np.clip(action[3], -1.0, 1.0))
        if self.ball_vx < 0:
            target_y = self.predict_ball_y_at_x(self.agent_x)
            if self.required_landing_side == "agent" and self.shot_landed:
                target_x = np.clip(self.ball_x - 24.0 + self.attack_intent * 28.0, self.config.agent_x_min, self.config.agent_x_max)
            else:
                target_x = self.config.agent_x_min + 78.0 + max(self.attack_intent, 0.0) * 28.0
            desired_angle = np.clip((target_y - self.agent_y) / 130.0 + self.attack_intent * 0.32, -1.0, 1.0) * self.config.max_paddle_angle
        else:
            target_y = self.config.table_y - 95
            target_x = self.config.agent_x_min + 70.0 + max(self.attack_intent, 0.0) * 24.0
            desired_angle = self.attack_intent * 0.28
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
        self.agent_min_x_seen = min(self.agent_min_x_seen, self.agent_x)
        self.agent_max_x_seen = max(self.agent_max_x_seen, self.agent_x)

    def _apply_opponent_policy(self) -> None:
        if self.ball_vx > 0:
            target_y = self.predict_ball_y_at_x(self.opponent_x)
            if self.required_landing_side == "opponent" and self.shot_landed:
                target_x = np.clip(self.ball_x + 24.0, self.config.opponent_x_min, self.config.opponent_x_max)
            else:
                target_x = self.config.opponent_x_max - 55
            difficulty = self._incoming_difficulty()
            level_resilience = 0.08 * self.opponent_level
            if difficulty > 0.72 + level_resilience:
                target_y += float(self.np_random.normal(0.0, 8.0 + 14.0 * difficulty))
                target_x -= 8.0 * difficulty
            desired_angle = -0.15 if self.ball_y < self.opponent_y else 0.15
            speed_scale = float(np.clip(1.12 - 0.28 * max(difficulty - 0.55, 0.0) + 0.03 * self.opponent_level, 0.78, 1.22))
        else:
            target_y = self.config.table_y - 95
            target_x = self.config.opponent_x_max - 55
            desired_angle = 0.0
            speed_scale = 1.0
        self.opponent_vy = float(np.clip(target_y - self.opponent_y, -self.config.opponent_speed * speed_scale, self.config.opponent_speed * speed_scale))
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

    def _bounce_from_paddle(self, player: str) -> None:
        if player == "agent":
            direction = 1
            paddle_x = self.agent_x
            paddle_y = self.agent_y
            paddle_vx = self.agent_vx
            paddle_vy = self.agent_vy
            angle = self.agent_angle
            attack = max(self.attack_intent, 0.0)
            landing_min = self.config.net_x + 70
            landing_max = self.opponent_x - 40
            base_landing_x = self.config.deep_landing_x + attack * 60.0
        else:
            direction = -1
            paddle_x = self.opponent_x
            paddle_y = self.opponent_y
            paddle_vx = self.opponent_vx
            paddle_vy = self.opponent_vy
            angle = self.opponent_angle
            attack = 0.0
            landing_min = self.agent_x + 125
            landing_max = self.config.net_x - 55
            base_landing_x = (landing_min + landing_max) / 2

        offset = float(np.clip((self.ball_y - paddle_y) / (self.config.paddle_height / 2), -1.0, 1.0))
        self.last_contact_quality = float(1.0 - min(abs(offset), 1.0))
        self.ball_x = paddle_x + direction * (self.config.paddle_width / 2 + self.config.ball_radius + 1)
        speed_factor = 0.96 + 0.08 * attack if player == "agent" else 0.93
        speed_cap = 9.1 if player == "agent" else 8.4
        speed = min(
            max(abs(self.ball_vx) * speed_factor + 0.25 + abs(paddle_vx) * 0.08, self.config.ball_speed_x_min),
            speed_cap,
        )
        self.ball_vx = direction * speed
        landing_x = base_landing_x + direction * angle * 110.0 + offset * 35.0 + paddle_vx * 2.4
        landing_x = float(np.clip(landing_x, landing_min, landing_max))
        self.ball_vy = self._aimed_vertical_velocity(landing_x, self.config.table_y - self.config.ball_radius)
        if player == "agent":
            self.ball_vy -= 0.65 + attack * 0.45 + max(0.0, abs(angle) - 0.15) * 0.55
        else:
            self.ball_vy -= 1.05 + max(0.0, abs(angle) - 0.15) * 0.35
        spin_delta = angle * (4.8 + attack * 2.2) + paddle_vy * 0.08 - offset * 2.0 + attack * 0.8
        self.ball_spin = float(np.clip(self.ball_spin * 0.42 + spin_delta, -self.config.max_spin, self.config.max_spin))

    def _incoming_difficulty(self) -> float:
        speed = float(np.hypot(self.ball_vx, self.ball_vy))
        deep = 1.0 if self.last_hitter == "agent" and self.ball_x >= self.config.deep_landing_x else 0.0
        return float(np.clip((speed - 6.2) / 5.0 + abs(self.ball_spin) / 8.5 + deep * 0.28, 0.0, 1.4))

    def _maybe_force_opponent_error(self, legal_landing: bool) -> None:
        if not legal_landing or self.point_winner is not None or self.last_hitter != "agent":
            return
        if not (self.last_agent_attack or self.last_agent_deep_landing) or self.rally_length < 2:
            return
        difficulty = self._incoming_difficulty()
        level_resilience = 0.035 * self.opponent_level
        probability = self.config.opponent_base_error + self.config.opponent_attack_error_scale * max(difficulty - 0.55 - level_resilience, 0.0)
        if self.np_random.random() < min(probability, 0.16):
            self._award_point("agent", "forced_error")

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
        reward = super()._reward(previous_distance, action[:3], agent_hit, legal_landing, agent_score, agent_miss, rally_success)
        reward += 0.02 * abs(self.agent_vx)
        if agent_hit and self.attack_intent > 0.25:
            reward += 0.12 + 0.08 * min(abs(self.ball_spin), self.config.max_spin)
        if agent_hit:
            reward += 0.03 * min(self.rally_length, 16)
        if legal_landing and self.last_hitter == "agent" and self.last_agent_deep_landing:
            reward += self.config.attack_landing_reward
        if agent_score and (self.last_agent_attack or self.last_agent_deep_landing):
            reward += self.config.attack_score_bonus
        if not agent_score and self.point_reason == "forced_error":
            reward += self.config.attack_score_bonus
        return float(reward)

    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()
        extras = np.array(
            [
                self.attack_intent,
                (self.agent_max_x_seen - self.agent_min_x_seen) / max(self.config.agent_x_max - self.config.agent_x_min, 1.0) * 2 - 1,
            ],
            dtype=np.float32,
        )
        return np.clip(np.concatenate([base, extras]), -1.0, 1.0).astype(np.float32)

    def _get_info(self, agent_hit: bool, agent_score: bool, agent_miss: bool, rally_success: bool, legal_landing: bool) -> dict:
        info = super()._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        info.update(
            {
                "stage": 8,
                "attack_attempts": self.attack_attempts,
                "attack_landings": self.attack_landings,
                "attack_scores": self.attack_scores,
                "attack_success": self.point_reason == "forced_error"
                or (agent_score and (self.last_agent_attack or self.last_agent_deep_landing)),
                "paddle_x_range": self.agent_max_x_seen - self.agent_min_x_seen,
                "opponent_level": self.opponent_level,
            }
        )
        return info
