from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pingpong_rl.envs.self_play_variety_env import SelfPlayVarietyConfig, SelfPlayVarietyEnv


@dataclass(frozen=True)
class LeagueSelfPlayConfig(SelfPlayVarietyConfig):
    max_steps: int = 4500
    short_rally_target: int = 6
    short_win_penalty: float = 14.0
    rally_progress_reward: float = 0.22
    opponent_return_reward: float = 0.80
    sustained_win_bonus: float = 4.0
    opponent_reaction_noise: float = 0.0
    opponent_safety_angle_scale: float = 0.58
    opponent_attack_scale: float = 0.45


class LeagueSelfPlayEnv(SelfPlayVarietyEnv):
    """Stage 13: multi-opponent league self-play with anti-short-rally rewards."""

    def __init__(self, render_mode: str | None = None, config: LeagueSelfPlayConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or LeagueSelfPlayConfig())

    def _bounce_from_paddle(self, player: str) -> None:
        super()._bounce_from_paddle(player)
        if player != "opponent":
            return

        landing_min = max(self.config.table_left + 42.0, self.agent_x + 135.0)
        landing_max = self.config.net_x - 72.0
        if landing_min >= landing_max:
            landing_min = self.config.table_left + 50.0
            landing_max = self.config.net_x - 78.0
        landing_x = float(np.clip((landing_min + landing_max) / 2.0, landing_min, landing_max))
        speed = float(np.clip(abs(self.ball_vx), 6.8, 8.2))
        self.ball_vx = -speed
        self.ball_vy = self._aimed_vertical_velocity(landing_x, self.config.table_y - self.config.ball_radius) - 1.35
        self.ball_spin = float(np.clip(self.ball_spin * 0.35 - 0.4, -self.config.max_spin, self.config.max_spin))

    def _get_info(self, agent_hit: bool, agent_score: bool, agent_miss: bool, rally_success: bool, legal_landing: bool) -> dict:
        info = super()._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        info.update(
            {
                "stage": 13,
                "league_enabled": True,
                "short_rally_target": self.self_play_config.short_rally_target,
            }
        )
        return info


class RedLeagueSelfPlayEnv(LeagueSelfPlayEnv):
    """Stage 13 role-swap training: external actions control the red paddle."""

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        return self._get_mirrored_obs_for_history_opponent(), self._get_info(False, False, False, False, False)

    def step(self, action):
        self.steps += 1
        action_arr = np.asarray(action, dtype=np.float32)
        previous_distance = self._distance_to_red_target()
        self._apply_blue_model_policy()
        self._apply_model_opponent_action(action_arr)
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
        red_score = agent_miss
        red_miss = agent_score
        rally_success = self.legal_landings >= self.config.target_rally_length + 2
        reward = self._red_reward(previous_distance, action_arr, opponent_hit, legal_landing, red_score, red_miss, rally_success)
        terminated = bool(agent_score or agent_miss or rally_success)
        truncated = bool(self.steps >= self.config.max_steps and not terminated)
        info = self._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        info.update(
            {
                "controlled_side": "red",
                "red_score": red_score,
                "red_miss": red_miss,
                "red_hits": self.opponent_hits,
                "blue_hits": self.agent_hits,
            }
        )

        if self.render_mode == "human":
            self.render()
        return self._get_mirrored_obs_for_history_opponent(), reward, terminated, truncated, info

    def _apply_blue_model_policy(self) -> None:
        self._load_history_opponent()
        if self.history_opponent is None:
            return super()._apply_agent_action(np.zeros(self.action_space.shape, dtype=np.float32))

        action, _ = self.history_opponent.predict(self._get_obs(), deterministic=self.self_play_config.opponent_deterministic)
        super()._apply_agent_action(np.asarray(action, dtype=np.float32))
        self.model_opponent_steps += 1

    def _distance_to_red_target(self) -> float:
        if self.ball_vx > 0:
            target_y = self.predict_ball_y_at_x(self.opponent_x)
        else:
            target_y = self.config.table_y - 102
        return abs(target_y - self.opponent_y) / max(self.config.height, 1.0)

    def _red_reward(
        self,
        previous_distance: float,
        action: np.ndarray,
        red_hit: bool,
        legal_landing: bool,
        red_score: bool,
        red_miss: bool,
        rally_success: bool,
    ) -> float:
        reward = 0.0
        if self.ball_vx > 0:
            reward += self.config.shaping_scale * (previous_distance - self._distance_to_red_target())
        if red_hit:
            reward += self.config.hit_reward
            reward += self.self_play_config.rally_progress_reward * min(self.rally_length, 12)
        if legal_landing and self.last_hitter == "opponent":
            reward += self.config.legal_landing_reward
            reward += self.self_play_config.opponent_return_reward
        elif legal_landing and self.last_hitter == "agent":
            reward += 0.15
        if red_score:
            shortfall = max(self.self_play_config.short_rally_target - self.rally_length, 0)
            if shortfall > 0:
                reward -= self.self_play_config.short_win_penalty * shortfall / self.self_play_config.short_rally_target
            else:
                reward += self.config.score_reward + self.self_play_config.sustained_win_bonus
        if red_miss:
            reward += self.config.miss_penalty
        if rally_success:
            reward += self.config.rally_success_reward + self.self_play_config.sustained_win_bonus
        reward -= self.config.move_penalty * float(np.sum(np.square(action)))
        return float(reward)
