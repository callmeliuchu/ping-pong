from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pingpong_rl.envs.self_play_variety_env import SelfPlayVarietyConfig, SelfPlayVarietyEnv


@dataclass(frozen=True)
class LeagueSelfPlayConfig(SelfPlayVarietyConfig):
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
