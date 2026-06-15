"""Gymnasium environments for the ping pong RL project."""

from pingpong_rl.envs.catch_env import CatchEnv
from pingpong_rl.envs.advanced_strokes_env import AdvancedStrokesEnv
from pingpong_rl.envs.compact_technique_env import CompactTechniqueEnv
from pingpong_rl.envs.competitive_env import CompetitiveRealisticEnv
from pingpong_rl.envs.gravity_env import GravityPingPongEnv
from pingpong_rl.envs.pong_env import PongEnv, make_pong_config
from pingpong_rl.envs.realistic_env import RealisticPingPongEnv
from pingpong_rl.envs.self_play_variety_env import SelfPlayVarietyEnv
from pingpong_rl.envs.variety_technique_env import VarietyTechniqueEnv

__all__ = [
    "CatchEnv",
    "AdvancedStrokesEnv",
    "CompetitiveRealisticEnv",
    "CompactTechniqueEnv",
    "GravityPingPongEnv",
    "PongEnv",
    "RealisticPingPongEnv",
    "SelfPlayVarietyEnv",
    "VarietyTechniqueEnv",
    "make_pong_config",
]
