from __future__ import annotations

from dataclasses import dataclass

from pingpong_rl.envs.robot_arm_champion_attack_env import (
    RobotArmChampionAttackConfig,
    RobotArmChampionAttackEnv,
)


@dataclass(frozen=True)
class RobotArmGrandChampionAttackConfig(RobotArmChampionAttackConfig):
    """Stage 21: beat the current champion while preserving clean finish pressure."""

    forced_finish_reward: float = 6.2
    timely_finish_reward: float = 4.0
    stalemate_penalty: float = 7.0
    late_stalemate_pressure_penalty: float = 3.2
    timely_finish_max_rally: int = 9
    clean_combo_score_reward: float = 5.8
    adaptive_score_reward: float = 4.6


class RobotArmGrandChampionAttackEnv(RobotArmChampionAttackEnv):
    """Stage 21: grand champion league with the Stage 20 champion in the pool."""

    def __init__(self, render_mode: str | None = None, config: RobotArmGrandChampionAttackConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or RobotArmGrandChampionAttackConfig())
        self.config: RobotArmGrandChampionAttackConfig
        self.robot_arm_grand_champion_attack_enabled = True

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
                "stage": 21,
                "robot_arm_grand_champion_attack_enabled": True,
            }
        )
        return info
