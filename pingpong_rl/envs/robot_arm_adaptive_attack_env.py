from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pingpong_rl.envs.robot_arm_attack_league_env import RobotArmAttackLeagueConfig, RobotArmAttackLeagueEnv


PressureProfile = tuple[str, float, float]


@dataclass(frozen=True)
class RobotArmAdaptiveAttackConfig(RobotArmAttackLeagueConfig):
    """Stage 18: train attack that works across different red-side pressure tolerance profiles."""

    pressure_profiles: tuple[PressureProfile, ...] = field(
        default_factory=lambda: (
            ("resistant", 145.0, 0.60),
            ("balanced", 150.0, 0.62),
            ("fragile", 155.0, 0.63),
        )
    )
    pressure_profile_index: int = -1
    deep_attack_reward: float = 0.34
    wide_attack_reward: float = 0.32
    combo_attack_reward: float = 0.72
    adaptive_score_reward: float = 2.8
    deep_landing_x: float = 585.0
    wide_landing_gap: float = 12.0


class RobotArmAdaptiveAttackEnv(RobotArmAttackLeagueEnv):
    """Stage 18: attack league with randomized red pressure profiles and placement shaping."""

    def __init__(self, render_mode: str | None = None, config: RobotArmAdaptiveAttackConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or RobotArmAdaptiveAttackConfig())
        self.config: RobotArmAdaptiveAttackConfig
        self.robot_arm_adaptive_attack_enabled = True
        self.current_pressure_profile_name = "balanced"
        self.current_pressure_offset = self.config.pressure_tracking_offset
        self.current_pressure_slowdown = self.config.pressure_slowdown
        self.deep_attack_landings = 0
        self.wide_attack_landings = 0
        self.combo_attack_landings = 0
        self.adaptive_clean_scores = 0
        self.last_pressure_tracking_error = 0.0
        self.max_pressure_tracking_error = 0.0

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self._select_pressure_profile()
        self.deep_attack_landings = 0
        self.wide_attack_landings = 0
        self.combo_attack_landings = 0
        self.adaptive_clean_scores = 0
        self.last_pressure_tracking_error = 0.0
        self.max_pressure_tracking_error = 0.0
        info = self._get_info(False, False, False, False, False)
        return obs, info

    def _select_pressure_profile(self) -> None:
        profiles = self.config.pressure_profiles
        if not profiles:
            self.current_pressure_profile_name = "default"
            self.current_pressure_offset = self.config.pressure_tracking_offset
            self.current_pressure_slowdown = self.config.pressure_slowdown
            return
        if 0 <= self.config.pressure_profile_index < len(profiles):
            profile = profiles[self.config.pressure_profile_index]
        else:
            profile = profiles[int(self.np_random.integers(0, len(profiles)))]
        self.current_pressure_profile_name = profile[0]
        self.current_pressure_offset = float(profile[1])
        self.current_pressure_slowdown = float(profile[2])

    def _pressure_tracking_offset(self) -> float:
        return self.current_pressure_offset

    def _pressure_slowdown(self) -> float:
        return self.current_pressure_slowdown

    def _apply_model_opponent_action(self, action: np.ndarray) -> None:
        pressured = self.ball_vx > 0 and self.last_hitter == "agent" and self.last_attack_pressure >= self.config.pressure_threshold
        super()._apply_model_opponent_action(action)
        if pressured:
            self.last_pressure_tracking_error = self.opponent_tracking_error
            self.max_pressure_tracking_error = max(self.max_pressure_tracking_error, self.opponent_tracking_error)

    def _placement_flags(self) -> tuple[bool, bool, bool]:
        deep = self.ball_x >= self.config.deep_landing_x
        wide = self.last_pressure_tracking_error >= self.config.wide_landing_gap
        combo = deep and wide and self.last_attack_pressure >= self.config.pressure_threshold
        return bool(deep), bool(wide), bool(combo)

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
        reward = super()._reward(previous_distance, action, agent_hit, legal_landing, agent_score, agent_miss, rally_success)
        if legal_landing and self.last_hitter == "agent":
            deep, wide, combo = self._placement_flags()
            self.deep_attack_landings += int(deep)
            self.wide_attack_landings += int(wide)
            self.combo_attack_landings += int(combo)
            reward += self.config.deep_attack_reward * int(deep)
            reward += self.config.wide_attack_reward * int(wide)
            reward += self.config.combo_attack_reward * int(combo)
        if agent_score and self.point_reason != "wrong_side_landing" and self.combo_attack_landings > 0:
            self.adaptive_clean_scores += 1
            reward += self.config.adaptive_score_reward
        return float(reward)

    def _get_info(
        self,
        agent_hit: bool,
        agent_score: bool,
        agent_miss: bool,
        rally_success: bool,
        legal_landing: bool,
    ) -> dict:
        info = super()._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        info.update(
            {
                "stage": 18,
                "robot_arm_adaptive_attack_enabled": True,
                "pressure_profile": self.current_pressure_profile_name,
                "pressure_profile_offset": self.current_pressure_offset,
                "pressure_profile_slowdown": self.current_pressure_slowdown,
                "deep_attack_landings": self.deep_attack_landings,
                "wide_attack_landings": self.wide_attack_landings,
                "combo_attack_landings": self.combo_attack_landings,
                "adaptive_clean_scores": self.adaptive_clean_scores,
                "last_pressure_tracking_error": self.last_pressure_tracking_error,
                "max_pressure_tracking_error": self.max_pressure_tracking_error,
            }
        )
        return info
