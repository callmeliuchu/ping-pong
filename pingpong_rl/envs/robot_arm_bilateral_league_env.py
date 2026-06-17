from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pingpong_rl.envs.robot_arm_grand_champion_attack_env import (
    RobotArmGrandChampionAttackConfig,
    RobotArmGrandChampionAttackEnv,
)


@dataclass(frozen=True)
class RobotArmBilateralLeagueConfig(RobotArmGrandChampionAttackConfig):
    """Stage 22: blue/red champion candidates train and challenge each other."""

    blue_model_paths: tuple[str, ...] = field(default_factory=tuple)
    bilateral_score_margin: float = 0.55
    red_score_reward: float = 9.0
    red_miss_penalty: float = -7.0
    red_legal_landing_reward: float = 1.5
    red_hit_reward: float = 1.7
    red_rally_reward: float = 0.18
    red_forced_finish_reward: float = 5.2
    red_stalemate_penalty: float = 5.4


class RobotArmBilateralLeagueEnv(RobotArmGrandChampionAttackEnv):
    """Stage 22 blue-side environment with explicit bilateral-league metadata."""

    def __init__(self, render_mode: str | None = None, config: RobotArmBilateralLeagueConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or RobotArmBilateralLeagueConfig())
        self.config: RobotArmBilateralLeagueConfig
        self.robot_arm_bilateral_league_enabled = True
        self.controlled_side = "blue"

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
                "stage": 22,
                "controlled_side": "blue",
                "robot_arm_bilateral_league_enabled": True,
                "blue_score": agent_score,
                "red_score": agent_miss,
                "blue_hits": self.agent_hits,
                "red_hits": self.opponent_hits,
            }
        )
        return info


class RedRobotArmBilateralLeagueEnv(RobotArmBilateralLeagueEnv):
    """Stage 22 role-swap environment: external actions control the red robot arm."""

    def __init__(self, render_mode: str | None = None, config: RobotArmBilateralLeagueConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or RobotArmBilateralLeagueConfig())
        self.blue_policy = None
        self.blue_policy_path = ""
        self.blue_policy_loaded = False
        self.controlled_side = "red"

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        self.blue_policy = None
        self.blue_policy_path = ""
        self.blue_policy_loaded = False
        self._load_blue_policy()
        return self._get_mirrored_obs_for_opponent(), self._get_info(False, False, False, False, False)

    def _load_blue_policy(self) -> None:
        if self.blue_policy is not None or not self.config.blue_model_paths:
            return

        candidates = [Path(path) for path in self.config.blue_model_paths]
        existing = [path for path in candidates if path.exists() or path.with_suffix(".zip").exists()]
        if not existing:
            return
        selected = existing[int(self.np_random.integers(0, len(existing)))]
        from stable_baselines3 import PPO

        self.blue_policy = PPO.load(selected)
        self.blue_policy_path = str(selected)
        self.blue_policy_loaded = True

    def _apply_blue_policy(self) -> None:
        self._load_blue_policy()
        if self.blue_policy is None:
            return super()._apply_agent_action(np.zeros(self.action_space.shape, dtype=np.float32))
        action, _ = self.blue_policy.predict(self._get_obs(), deterministic=self.config.opponent_deterministic)
        super()._apply_agent_action(np.asarray(action, dtype=np.float32))

    def step(self, action):
        self.steps += 1
        action_arr = np.asarray(action, dtype=np.float32)
        previous_distance = self._red_distance_to_target()
        self._apply_blue_policy()
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

        blue_score = self.point_winner == "agent"
        red_score = self.point_winner == "opponent"
        rally_success = self.legal_landings >= self.config.target_rally_length + 2
        reward = self._red_reward(previous_distance, action_arr, opponent_hit, legal_landing, red_score, blue_score, rally_success)
        terminated = bool(blue_score or red_score or rally_success)
        truncated = bool(self.steps >= self.config.max_steps and not terminated)
        info = self._get_info(agent_hit, blue_score, red_score, rally_success, legal_landing)

        if self.render_mode == "human":
            self.render()
        return self._get_mirrored_obs_for_opponent(), reward, terminated, truncated, info

    def _red_distance_to_target(self) -> float:
        if self.ball_vx > 0:
            target_y = self.predict_ball_y_at_x(self.opponent_x)
        else:
            target_y = self.config.table_y - 98.0
        return abs(target_y - self.opponent_y) / max(self.config.height, 1.0)

    def _red_forced_finish(self, red_score: bool) -> bool:
        return bool(
            red_score
            and self.point_reason in {"receiver_missed", "second_bounce"}
            and self.rally_length >= self.config.clean_attack_rally_target
        )

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
            reward += self.config.shaping_scale * (previous_distance - self._red_distance_to_target())
        reward -= self.config.move_penalty * float(np.sum(np.square(action)))
        reward -= self.config.joint_center_penalty * float(np.mean(np.square(self.opponent_joint_angles)))
        if red_hit:
            reward += self.config.red_hit_reward
            reward += self.config.red_rally_reward * min(self.rally_length, 16)
            if self.red_last_stroke_type in {"loop", "drive", "chop"}:
                reward += self.config.technique_reward
        if legal_landing and self.last_hitter == "opponent":
            reward += self.config.red_legal_landing_reward
            if self.red_last_stroke_type == "loop":
                reward += self.config.topspin_landing_reward
            if abs(self.ball_vx) >= 7.3:
                reward += self.config.drive_landing_reward
        elif legal_landing and self.last_hitter == "agent":
            reward += 0.25
        if red_score:
            reward += self.config.red_score_reward
            if self._red_forced_finish(red_score):
                reward += self.config.red_forced_finish_reward
        if red_miss:
            reward += self.config.red_miss_penalty
        if rally_success:
            reward -= self.config.red_stalemate_penalty
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
                "controlled_side": "red",
                "blue_policy_loaded": self.blue_policy_loaded,
                "blue_policy_path": self.blue_policy_path,
                "blue_score": agent_score,
                "red_score": agent_miss,
                "blue_hits": self.agent_hits,
                "red_hits": self.opponent_hits,
            }
        )
        return info
