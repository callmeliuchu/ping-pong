from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces

from pingpong_rl.envs.realistic_env import RealisticPingPongConfig, RealisticPingPongEnv


@dataclass(frozen=True)
class RobotArmPingPongConfig(RealisticPingPongConfig):
    """Stage 14: true joint-controlled paddle instead of direct paddle commands."""

    max_steps: int = 3600
    target_rally_length: int = 14
    max_spin: float = 16.0
    spin_lift: float = 0.016
    spin_bounce_coupling: float = 0.070
    max_ball_speed: float = 17.0
    agent_x_min: float = 45.0
    agent_x_max: float = 330.0
    opponent_x_min: float = 570.0
    opponent_x_max: float = 855.0
    arm_upper_length: float = 136.0
    arm_forearm_length: float = 132.0
    arm_hand_length: float = 22.0
    arm_shoulder_speed: float = 0.055
    arm_elbow_speed: float = 0.070
    arm_wrist_speed: float = 0.090
    shoulder_min: float = -1.45
    shoulder_max: float = 0.95
    elbow_min: float = -1.75
    elbow_max: float = 1.75
    wrist_min: float = -0.95
    wrist_max: float = 0.95
    opponent_arm_speed_scale: float = 1.45
    end_effector_reward: float = 0.018
    technique_reward: float = 0.30
    topspin_landing_reward: float = 0.64
    backspin_landing_reward: float = 0.06
    drive_landing_reward: float = 0.48
    loop_arc_reward: float = 0.44
    joint_center_penalty: float = 0.002
    move_penalty: float = 0.004


class RobotArmPingPongEnv(RealisticPingPongEnv):
    """Stage 14: train shoulder, elbow, and wrist velocities to hit the ball."""

    def __init__(self, render_mode: str | None = None, config: RobotArmPingPongConfig | None = None):
        super().__init__(render_mode=render_mode, config=config or RobotArmPingPongConfig())
        self.config: RobotArmPingPongConfig
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(24,), dtype=np.float32)
        self.agent_arm_base = (self.config.table_left - 64.0, self.config.table_y + 76.0)
        self.opponent_arm_base = (self.config.table_right + 64.0, self.config.table_y + 76.0)
        self.agent_joint_angles = np.zeros(3, dtype=np.float32)
        self.agent_joint_velocities = np.zeros(3, dtype=np.float32)
        self.opponent_joint_angles = np.zeros(3, dtype=np.float32)
        self.opponent_joint_velocities = np.zeros(3, dtype=np.float32)
        self.agent_wrist_x = 0.0
        self.agent_wrist_y = 0.0
        self.opponent_wrist_x = 0.0
        self.opponent_wrist_y = 0.0
        self.agent_tracking_error = 0.0
        self.opponent_tracking_error = 0.0
        self.ball_trail: list[tuple[float, float]] = []
        self.last_stroke_type = "none"
        self.loop_attempts = 0
        self.loop_landings = 0
        self.drive_attempts = 0
        self.drive_landings = 0
        self.chop_attempts = 0
        self.chop_landings = 0
        self.topspin_attempts = 0
        self.topspin_landings = 0
        self.backspin_attempts = 0
        self.backspin_landings = 0
        self.max_topspin = 0.0
        self.max_backspin = 0.0
        self.agent_shot_start_y = 0.0
        self.agent_shot_min_y = 0.0
        self.agent_shot_peak_arc = 0.0
        self.last_agent_action = np.zeros(3, dtype=np.float32)
        self.agent_last_drive_like = False
        self.agent_last_topspin_like = False
        self.agent_last_backspin_like = False
        self.agent_last_shot_speed = 0.0

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed, options=options)
        self.agent_joint_velocities[:] = 0.0
        self.opponent_joint_velocities[:] = 0.0
        agent_target = (self.config.table_left + 72.0, self.config.table_y - 98.0, -0.28)
        opponent_target = (self.config.table_right - 72.0, self.config.table_y - 98.0, 0.28)
        self.agent_joint_angles = self._ik_for_paddle("agent", *agent_target).astype(np.float32)
        self.opponent_joint_angles = self._ik_for_paddle("opponent", *opponent_target).astype(np.float32)
        self._sync_agent_from_arm()
        self._sync_opponent_from_arm()
        self.agent_tracking_error = self._distance_to_agent_target()
        self.opponent_tracking_error = 0.0
        self.ball_trail = [(self.ball_x, self.ball_y)]
        self.last_stroke_type = "none"
        self.loop_attempts = 0
        self.loop_landings = 0
        self.drive_attempts = 0
        self.drive_landings = 0
        self.chop_attempts = 0
        self.chop_landings = 0
        self.topspin_attempts = 0
        self.topspin_landings = 0
        self.backspin_attempts = 0
        self.backspin_landings = 0
        self.max_topspin = 0.0
        self.max_backspin = 0.0
        self.agent_shot_start_y = self.ball_y
        self.agent_shot_min_y = self.ball_y
        self.agent_shot_peak_arc = 0.0
        self.agent_last_drive_like = False
        self.agent_last_topspin_like = False
        self.agent_last_backspin_like = False
        self.agent_last_shot_speed = 0.0
        return self._get_obs(), self._get_info(False, False, False, False, False)

    def _apply_agent_action(self, action: np.ndarray) -> None:
        speeds = np.array(
            [self.config.arm_shoulder_speed, self.config.arm_elbow_speed, self.config.arm_wrist_speed],
            dtype=np.float32,
        )
        clipped = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.last_agent_action = clipped
        self.agent_joint_velocities = clipped * speeds
        self.agent_joint_angles = self._clip_joint_angles(self.agent_joint_angles + self.agent_joint_velocities)
        self._sync_agent_from_arm()
        self.agent_tracking_error = self._distance_to_agent_target()

    def _apply_opponent_policy(self) -> None:
        if self.ball_vx > 0:
            target_x = float(np.clip(self.ball_x + 34.0, self.config.opponent_x_min, self.config.opponent_x_max))
            target_y = self.predict_ball_y_at_x(target_x)
            desired_angle = 0.32 if self.ball_y < self.opponent_y else -0.08
        else:
            target_x = self.config.table_right - 72.0
            target_y = self.config.table_y - 98.0
            desired_angle = 0.20
        target_angles = self._ik_for_paddle("opponent", target_x, target_y, desired_angle)
        max_step = np.array(
            [self.config.arm_shoulder_speed, self.config.arm_elbow_speed, self.config.arm_wrist_speed],
            dtype=np.float32,
        ) * self.config.opponent_arm_speed_scale
        delta = np.clip(target_angles - self.opponent_joint_angles, -max_step, max_step)
        self.opponent_joint_velocities = delta.astype(np.float32)
        self.opponent_joint_angles = self._clip_joint_angles(self.opponent_joint_angles + self.opponent_joint_velocities)
        self._sync_opponent_from_arm()
        self.opponent_tracking_error = float(np.hypot(target_x - self.opponent_x, target_y - self.opponent_y))

    def _move_ball(self) -> None:
        self.ball_vy += self.config.gravity + self.config.spin_lift * self.ball_spin
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        speed = float(np.hypot(self.ball_vx, self.ball_vy))
        if speed > self.config.max_ball_speed:
            scale = self.config.max_ball_speed / speed
            self.ball_vx *= scale
            self.ball_vy *= scale
        self.ball_trail.append((self.ball_x, self.ball_y))
        if len(self.ball_trail) > 90:
            self.ball_trail = self.ball_trail[-90:]
        if self.last_hitter == "agent" and not self.shot_landed:
            self.agent_shot_min_y = min(self.agent_shot_min_y, self.ball_y)
            self.agent_shot_peak_arc = max(self.agent_shot_peak_arc, self.agent_shot_start_y - self.agent_shot_min_y)

    def _bounce_from_paddle(self, player: str) -> None:
        super()._bounce_from_paddle(player)
        if player != "agent":
            landing_x = float(
                np.clip(
                    self.config.table_left + 185.0 + 0.18 * (self.agent_x - self.config.agent_x_min),
                    self.config.table_left + 85.0,
                    self.config.net_x - 78.0,
                )
            )
            self.ball_vy = self._aimed_vertical_velocity(landing_x, self.config.table_y - self.config.ball_radius)
            self.ball_vy -= 0.30 + max(0.0, abs(self.opponent_angle) - 0.15) * 0.25
            self.ball_spin = float(np.clip(self.ball_spin * 0.72, -self.config.max_spin, self.config.max_spin))
            return

        wrist_brush = float(self.agent_joint_velocities[2] / max(self.config.arm_wrist_speed, 1e-6))
        upward_brush = float(max(-self.agent_vy / max(self.config.paddle_y_speed, 1.0), 0.0))
        downward_brush = float(max(self.agent_vy / max(self.config.paddle_y_speed, 1.0), 0.0))
        closed_face = float(max(-self.agent_angle / max(self.config.max_paddle_angle, 1e-6), 0.0))
        open_face = float(max(self.agent_angle / max(self.config.max_paddle_angle, 1e-6), 0.0))
        forward_swing = float(max(self.agent_vx / max(self.config.paddle_x_speed, 1.0), 0.0))
        center_contact = max(self.last_contact_quality, 0.0)

        topspin_intent = max(0.0, 0.45 * max(wrist_brush, 0.0) + 0.33 * upward_brush + 0.22 * closed_face)
        chop_intent = max(0.0, 0.48 * max(-wrist_brush, 0.0) + 0.34 * downward_brush + 0.18 * open_face)
        drive_intent = max(0.0, 0.62 * forward_swing + 0.38 * center_contact)

        if topspin_intent >= max(chop_intent, 0.18):
            spin_delta = 2.1 + 4.6 * topspin_intent
            self.ball_vx = min(self.ball_vx + 0.30 + 0.45 * forward_swing, self.config.max_ball_speed)
            self.ball_vy -= 0.16 + 0.42 * upward_brush
        elif drive_intent >= 0.58 and abs(self.ball_spin) < 4.8:
            spin_delta = 0.8 + 1.4 * closed_face - 0.6 * open_face
            self.ball_vx = min(self.ball_vx + 0.75 + 0.55 * forward_swing, self.config.max_ball_speed)
            self.ball_vy += 0.18 * downward_brush
        else:
            spin_delta = -1.3 - 3.4 * chop_intent
            self.ball_vx = max(self.ball_vx - 0.18 * chop_intent, self.config.ball_speed_x_min)
            self.ball_vy -= 0.08 * open_face
        self.ball_spin = float(np.clip(0.36 * self.ball_spin + spin_delta, -self.config.max_spin, self.config.max_spin))
        self.max_topspin = max(self.max_topspin, self.ball_spin)
        self.max_backspin = max(self.max_backspin, -self.ball_spin)
        self.agent_last_shot_speed = float(np.hypot(self.ball_vx, self.ball_vy))
        self.agent_last_drive_like = bool(abs(self.ball_vx) >= 7.3)
        self.agent_last_topspin_like = bool(self.ball_spin >= 2.0)
        self.agent_last_backspin_like = bool(self.ball_spin <= -1.5)
        self.last_stroke_type = self._classify_stroke()
        self.loop_attempts += int(self.last_stroke_type == "loop")
        self.drive_attempts += int(self.agent_last_drive_like)
        self.chop_attempts += int(self.last_stroke_type == "chop")
        self.topspin_attempts += int(self.agent_last_topspin_like)
        self.backspin_attempts += int(self.agent_last_backspin_like)
        self.agent_shot_start_y = self.ball_y
        self.agent_shot_min_y = self.ball_y
        self.agent_shot_peak_arc = 0.0

    def _classify_stroke(self) -> str:
        speed = abs(self.ball_vx)
        if self.ball_spin >= 3.0:
            return "loop"
        if speed >= 7.3:
            return "drive"
        if self.ball_spin <= -2.0:
            return "chop"
        return "block"

    def _sync_agent_from_arm(self) -> None:
        previous_x = self.agent_x
        previous_y = self.agent_y
        self.agent_x, self.agent_y, self.agent_angle, self.agent_wrist_x, self.agent_wrist_y = self._forward_kinematics(
            "agent", self.agent_joint_angles
        )
        self.agent_vx = self.agent_x - previous_x
        self.agent_vy = self.agent_y - previous_y

    def _sync_opponent_from_arm(self) -> None:
        previous_x = self.opponent_x
        previous_y = self.opponent_y
        (
            self.opponent_x,
            self.opponent_y,
            self.opponent_angle,
            self.opponent_wrist_x,
            self.opponent_wrist_y,
        ) = self._forward_kinematics("opponent", self.opponent_joint_angles)
        self.opponent_vx = self.opponent_x - previous_x
        self.opponent_vy = self.opponent_y - previous_y

    def _forward_kinematics(self, side: str, angles: np.ndarray) -> tuple[float, float, float, float, float]:
        base_x, base_y = self.agent_arm_base if side == "agent" else self.opponent_arm_base
        shoulder_x = base_x + (18.0 if side == "agent" else -18.0)
        shoulder_y = base_y - 46.0
        base_heading = 0.0 if side == "agent" else np.pi
        theta0 = base_heading + float(angles[0])
        theta1 = theta0 + float(angles[1])
        paddle_angle = float(angles[2])
        elbow_x = shoulder_x + self.config.arm_upper_length * float(np.cos(theta0))
        elbow_y = shoulder_y + self.config.arm_upper_length * float(np.sin(theta0))
        wrist_x = elbow_x + self.config.arm_forearm_length * float(np.cos(theta1))
        wrist_y = elbow_y + self.config.arm_forearm_length * float(np.sin(theta1))
        direction = 1.0 if side == "agent" else -1.0
        paddle_x = wrist_x + direction * self.config.arm_hand_length * float(np.cos(paddle_angle))
        paddle_y = wrist_y - self.config.arm_hand_length * float(np.sin(paddle_angle))
        return float(paddle_x), float(paddle_y), paddle_angle, wrist_x, wrist_y

    def _ik_for_paddle(self, side: str, paddle_x: float, paddle_y: float, paddle_angle: float) -> np.ndarray:
        base_x, base_y = self.agent_arm_base if side == "agent" else self.opponent_arm_base
        shoulder_x = base_x + (18.0 if side == "agent" else -18.0)
        shoulder_y = base_y - 46.0
        direction = 1.0 if side == "agent" else -1.0
        wrist_x = paddle_x - direction * self.config.arm_hand_length * float(np.cos(paddle_angle))
        wrist_y = paddle_y + self.config.arm_hand_length * float(np.sin(paddle_angle))
        dx = wrist_x - shoulder_x
        dy = wrist_y - shoulder_y
        distance = float(np.hypot(dx, dy))
        distance = max(1.0, min(distance, self.config.arm_upper_length + self.config.arm_forearm_length - 1.0))
        base_angle = float(np.arctan2(dy, dx))
        cos_offset = (
            self.config.arm_upper_length**2 + distance**2 - self.config.arm_forearm_length**2
        ) / (2.0 * self.config.arm_upper_length * distance)
        bend = 1.0 if side == "agent" else -1.0
        base_heading = 0.0 if side == "agent" else np.pi
        theta0 = base_angle + bend * float(np.arccos(np.clip(cos_offset, -1.0, 1.0)))
        elbow_x = shoulder_x + self.config.arm_upper_length * float(np.cos(theta0))
        elbow_y = shoulder_y + self.config.arm_upper_length * float(np.sin(theta0))
        theta1 = float(np.arctan2(wrist_y - elbow_y, wrist_x - elbow_x) - theta0)
        local_shoulder = self._wrap_angle(theta0 - base_heading)
        return self._clip_joint_angles(np.array([local_shoulder, self._wrap_angle(theta1), paddle_angle], dtype=np.float32))

    def _clip_joint_angles(self, angles: np.ndarray) -> np.ndarray:
        return np.array(
            [
                np.clip(angles[0], self.config.shoulder_min, self.config.shoulder_max),
                np.clip(angles[1], self.config.elbow_min, self.config.elbow_max),
                np.clip(angles[2], self.config.wrist_min, self.config.wrist_max),
            ],
            dtype=np.float32,
        )

    def _wrap_angle(self, angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

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
        reward -= self.config.end_effector_reward * min(self.agent_tracking_error / 100.0, 4.0)
        reward -= self.config.joint_center_penalty * float(np.mean(np.square(self.agent_joint_angles)))
        if agent_hit:
            reward += 0.10 * min(float(np.linalg.norm(self.agent_joint_velocities)) / self.config.arm_wrist_speed, 1.5)
            if self.last_stroke_type in {"loop", "drive", "chop"} or self.agent_last_drive_like:
                reward += self.config.technique_reward
            reward += 0.060 * min(max(self.ball_spin, 0.0), self.config.max_spin)
            reward += 0.006 * min(max(-self.ball_spin, 0.0), self.config.max_spin)
            reward += 0.045 * min(max(self.ball_vx - 7.0, 0.0), 4.0)
            if self.last_stroke_type == "chop" and self.rally_length < 4:
                reward -= 0.10
        if legal_landing and self.last_hitter == "agent":
            if self.last_stroke_type == "loop":
                reward += self.config.topspin_landing_reward
                if self.agent_shot_peak_arc >= 28.0:
                    reward += self.config.loop_arc_reward
            if self.agent_last_drive_like:
                reward += self.config.drive_landing_reward
            if self.last_stroke_type == "chop":
                reward += self.config.backspin_landing_reward
        return float(reward)

    def predict_ball_y_at_x(self, target_x: float) -> float:
        if abs(self.ball_vx) < 1e-6 or (target_x - self.ball_x) * self.ball_vx <= 0:
            return self.ball_y
        x = self.ball_x
        y = self.ball_y
        vx = self.ball_vx
        vy = self.ball_vy
        spin = self.ball_spin
        previous_x = x
        previous_y = y
        for _ in range(700):
            previous_x = x
            previous_y = y
            vy += self.config.gravity + self.config.spin_lift * spin
            x += vx
            y += vy
            if (target_x - previous_x) * (target_x - x) <= 0:
                span = x - previous_x
                alpha = 0.0 if abs(span) < 1e-6 else (target_x - previous_x) / span
                predicted_y = previous_y + alpha * (y - previous_y)
                return float(np.clip(predicted_y, self.config.table_y - 205, self.config.table_y - self.config.paddle_height / 2))
            over_table = self.config.table_left <= x <= self.config.table_right
            above_table = y + self.config.ball_radius >= self.config.table_y and vy > 0
            if over_table and above_table:
                y = self.config.table_y - self.config.ball_radius
                vy *= -self.config.bounce_damping
                vx += spin * self.config.spin_bounce_coupling
                vx *= 0.995
                spin *= 0.72
            net_top = self.config.table_y - self.config.net_height
            if abs(x - self.config.net_x) <= self.config.ball_radius and y + self.config.ball_radius >= net_top:
                break
        return float(np.clip(y, self.config.table_y - 205, self.config.table_y - self.config.paddle_height / 2))

    def _get_obs(self) -> np.ndarray:
        target_y = self.predict_ball_y_at_x(self.agent_x) if self.ball_vx < 0 else self.config.table_y - 98.0
        target_x = self.agent_x
        if self.ball_vx < 0:
            target_x = float(np.clip(self.ball_x - 36.0, self.config.agent_x_min, self.config.agent_x_max))
        angle_scales = np.array([self.config.shoulder_max, self.config.elbow_max, self.config.wrist_max], dtype=np.float32)
        speed_scales = np.array(
            [self.config.arm_shoulder_speed, self.config.arm_elbow_speed, self.config.arm_wrist_speed],
            dtype=np.float32,
        )
        obs = np.array(
            [
                self.ball_x / self.config.width * 2 - 1,
                self.ball_y / self.config.height * 2 - 1,
                self.ball_vx / self.config.max_ball_speed,
                self.ball_vy / self.config.max_ball_speed,
                self.ball_spin / self.config.max_spin,
                *(self.agent_joint_angles / angle_scales),
                *(self.agent_joint_velocities / speed_scales),
                self.agent_x / self.config.width * 2 - 1,
                self.agent_y / self.config.height * 2 - 1,
                self.agent_vx / max(self.config.paddle_x_speed, 1.0),
                self.agent_vy / max(self.config.paddle_y_speed, 1.0),
                self.agent_angle / self.config.max_paddle_angle,
                self.opponent_x / self.config.width * 2 - 1,
                self.opponent_y / self.config.height * 2 - 1,
                self.opponent_angle / self.config.max_paddle_angle,
                target_x / self.config.width * 2 - 1,
                target_y / self.config.height * 2 - 1,
                (target_x - self.agent_x) / self.config.width,
                (target_y - self.agent_y) / self.config.height,
                np.clip(self.rally_length / max(self.config.target_rally_length, 1), -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _get_info(
        self,
        agent_hit: bool,
        agent_score: bool,
        agent_miss: bool,
        rally_success: bool,
        legal_landing: bool,
    ) -> dict:
        info = super()._get_info(agent_hit, agent_score, agent_miss, rally_success, legal_landing)
        if legal_landing and self.last_hitter == "agent":
            self.loop_landings += int(self.last_stroke_type == "loop")
            self.drive_landings += int(self.agent_last_drive_like)
            self.chop_landings += int(self.last_stroke_type == "chop")
            self.topspin_landings += int(self.agent_last_topspin_like)
            self.backspin_landings += int(self.agent_last_backspin_like)
        info.update(
            {
                "stage": 14,
                "robot_arm_enabled": True,
                "last_stroke_type": self.last_stroke_type,
                "loop_attempts": self.loop_attempts,
                "loop_landings": self.loop_landings,
                "drive_attempts": self.drive_attempts,
                "drive_landings": self.drive_landings,
                "chop_attempts": self.chop_attempts,
                "chop_landings": self.chop_landings,
                "topspin_attempts": self.topspin_attempts,
                "topspin_landings": self.topspin_landings,
                "backspin_attempts": self.backspin_attempts,
                "backspin_landings": self.backspin_landings,
                "max_topspin": self.max_topspin,
                "max_backspin": self.max_backspin,
                "agent_shot_peak_arc": self.agent_shot_peak_arc,
                "agent_last_drive_like": self.agent_last_drive_like,
                "agent_last_topspin_like": self.agent_last_topspin_like,
                "agent_last_backspin_like": self.agent_last_backspin_like,
                "agent_last_shot_speed": self.agent_last_shot_speed,
                "agent_joint_angles": self.agent_joint_angles.tolist(),
                "agent_joint_velocities": self.agent_joint_velocities.tolist(),
                "agent_tracking_error": self.agent_tracking_error,
                "opponent_tracking_error": self.opponent_tracking_error,
                "agent_wrist_x": self.agent_wrist_x,
                "agent_wrist_y": self.agent_wrist_y,
                "opponent_wrist_x": self.opponent_wrist_x,
                "opponent_wrist_y": self.opponent_wrist_y,
            }
        )
        return info
