"""Gymnasium environments for the ping pong RL project."""

from pingpong_rl.envs.catch_env import CatchEnv
from pingpong_rl.envs.advanced_strokes_env import AdvancedStrokesEnv
from pingpong_rl.envs.compact_technique_env import CompactTechniqueEnv
from pingpong_rl.envs.competitive_env import CompetitiveRealisticEnv
from pingpong_rl.envs.gravity_env import GravityPingPongEnv
from pingpong_rl.envs.league_self_play_env import LeagueSelfPlayEnv
from pingpong_rl.envs.pong_env import PongEnv, make_pong_config
from pingpong_rl.envs.realistic_env import RealisticPingPongEnv
from pingpong_rl.envs.robot_arm_adaptive_attack_env import RobotArmAdaptiveAttackEnv
from pingpong_rl.envs.robot_arm_league_env import RobotArmLeagueEnv
from pingpong_rl.envs.robot_arm_attack_league_env import RobotArmAttackLeagueEnv
from pingpong_rl.envs.robot_arm_clean_adaptive_attack_env import RobotArmCleanAdaptiveAttackEnv
from pingpong_rl.envs.robot_arm_env import RobotArmPingPongEnv
from pingpong_rl.envs.robot_arm_tactical_league_env import RobotArmTacticalLeagueEnv
from pingpong_rl.envs.self_play_variety_env import SelfPlayVarietyEnv
from pingpong_rl.envs.variety_technique_env import VarietyTechniqueEnv

__all__ = [
    "CatchEnv",
    "AdvancedStrokesEnv",
    "CompetitiveRealisticEnv",
    "CompactTechniqueEnv",
    "GravityPingPongEnv",
    "LeagueSelfPlayEnv",
    "PongEnv",
    "RealisticPingPongEnv",
    "RobotArmAdaptiveAttackEnv",
    "RobotArmLeagueEnv",
    "RobotArmAttackLeagueEnv",
    "RobotArmCleanAdaptiveAttackEnv",
    "RobotArmPingPongEnv",
    "RobotArmTacticalLeagueEnv",
    "SelfPlayVarietyEnv",
    "VarietyTechniqueEnv",
    "make_pong_config",
]
