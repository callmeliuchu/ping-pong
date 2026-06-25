from __future__ import annotations

from dataclasses import dataclass

from pingpong_rl.envs.robot_arm_red_scoring_bilateral_env import (
    RobotArmRedScoringBilateralConfig,
    RobotArmRedScoringBilateralEnv,
)


@dataclass(frozen=True)
class RobotArmMirrorSelfPlayConfig(RobotArmRedScoringBilateralConfig):
    """Stage 24: left champion trains against its latest mirrored copy."""


class RobotArmMirrorSelfPlayEnv(RobotArmRedScoringBilateralEnv):
    """Stage 24 left-side environment using the current left champion as right mirror opponent."""

    def __init__(self, render_mode: str | None = None, config: RobotArmMirrorSelfPlayConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or RobotArmMirrorSelfPlayConfig())
        self.config: RobotArmMirrorSelfPlayConfig
        self.robot_arm_mirror_selfplay_enabled = True

    def _get_info(
        self,
        agent_hit: bool,
        agent_score: bool,
        agent_miss: bool,
        rally_success: bool,
        legal_landing: bool,
    ) -> dict:
        info = super()._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        info.update({"stage": 24, "robot_arm_mirror_selfplay_enabled": True})
        return info
