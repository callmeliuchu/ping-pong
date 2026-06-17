from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pingpong_rl.envs.robot_arm_clean_adaptive_attack_env import (
    RobotArmCleanAdaptiveAttackConfig,
    RobotArmCleanAdaptiveAttackEnv,
)


@dataclass(frozen=True)
class RobotArmChampionAttackConfig(RobotArmCleanAdaptiveAttackConfig):
    """Stage 20: clean attack finishing against the current champion pool."""

    forced_finish_reward: float = 5.4
    timely_finish_reward: float = 3.2
    stalemate_penalty: float = 5.8
    late_stalemate_pressure_penalty: float = 2.6
    timely_finish_min_rally: int = 4
    timely_finish_max_rally: int = 10
    clean_combo_score_reward: float = 5.2
    adaptive_score_reward: float = 4.2


class RobotArmChampionAttackEnv(RobotArmCleanAdaptiveAttackEnv):
    """Stage 20: champion-pool policy that should convert pressure into clean finishes."""

    def __init__(self, render_mode: str | None = None, config: RobotArmChampionAttackConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or RobotArmChampionAttackConfig())
        self.config: RobotArmChampionAttackConfig
        self.robot_arm_champion_attack_enabled = True
        self.forced_finish_scores = 0
        self.timely_finish_scores = 0
        self.stalemate_finishes = 0

    def reset(self, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        self.forced_finish_scores = 0
        self.timely_finish_scores = 0
        self.stalemate_finishes = 0
        info = self._get_info(False, False, False, False, False)
        return obs, info

    def _is_clean_combo_score(self, agent_score: bool) -> bool:
        return bool(
            agent_score
            and self.point_reason != "wrong_side_landing"
            and self.combo_attack_landings > 0
            and self.rally_length >= self.config.clean_attack_rally_target
        )

    def _is_forced_finish(self, agent_score: bool) -> bool:
        return bool(self._is_clean_combo_score(agent_score) and self.point_reason in {"receiver_missed", "second_bounce"})

    def _is_timely_finish(self, agent_score: bool) -> bool:
        return bool(
            self._is_forced_finish(agent_score)
            and self.config.timely_finish_min_rally <= self.rally_length <= self.config.timely_finish_max_rally
        )

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
        if self._is_forced_finish(agent_score):
            reward += self.config.forced_finish_reward
        if self._is_timely_finish(agent_score):
            reward += self.config.timely_finish_reward
        if rally_success and not agent_score:
            reward -= self.config.stalemate_penalty
            if self.max_attack_pressure >= self.config.pressure_threshold:
                reward -= self.config.late_stalemate_pressure_penalty
        return float(reward)

    def _get_info(
        self,
        agent_hit: bool,
        agent_score: bool,
        agent_miss: bool,
        rally_success: bool,
        legal_landing: bool,
    ) -> dict:
        if self._is_forced_finish(agent_score):
            self.forced_finish_scores += 1
        if self._is_timely_finish(agent_score):
            self.timely_finish_scores += 1
        if rally_success and not agent_score:
            self.stalemate_finishes += 1
        info = super()._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        info.update(
            {
                "stage": 20,
                "robot_arm_champion_attack_enabled": True,
                "forced_finish_scores": self.forced_finish_scores,
                "timely_finish_scores": self.timely_finish_scores,
                "stalemate_finishes": self.stalemate_finishes,
            }
        )
        return info
