from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class OpponentConfig:
    kind: str = "scripted"
    speed: float = 4.0
    model_path: str | None = None
    random_action_probability: float = 0.0


def scripted_action(ball_y: float, paddle_y: float, dead_zone: float = 4.0) -> int:
    if ball_y < paddle_y - dead_zone:
        return 1
    if ball_y > paddle_y + dead_zone:
        return 2
    return 0


def random_action(rng: np.random.Generator) -> int:
    return int(rng.integers(0, 3))


def predict_y_at_x(
    ball_x: float,
    ball_y: float,
    ball_vx: float,
    ball_vy: float,
    target_x: float,
    height: float,
    ball_radius: float,
) -> float:
    if abs(ball_vx) < 1e-6:
        return ball_y

    steps = (target_x - ball_x) / ball_vx
    if steps <= 0:
        return ball_y

    top = ball_radius
    bottom = height - ball_radius
    span = bottom - top
    raw_y = ball_y + ball_vy * steps
    folded = (raw_y - top) % (2 * span)
    if folded > span:
        folded = 2 * span - folded
    return float(top + folded)


class ModelOpponent:
    def __init__(self, model_path: str | Path):
        from stable_baselines3 import PPO

        self.model = PPO.load(model_path)

    def predict(self, mirrored_obs: np.ndarray) -> int:
        action, _ = self.model.predict(mirrored_obs, deterministic=True)
        return int(action)
