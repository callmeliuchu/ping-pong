from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from pingpong_rl.envs import RobotArmPingPongEnv
from train.eval_utils import ROOT, print_metrics, write_metrics


def evaluate_robot_arm_model(
    model_path: Path,
    episodes: int,
    render: bool = False,
    seed: int = 0,
) -> dict:
    env = RobotArmPingPongEnv(render_mode="human" if render else None)
    model = PPO.load(model_path)
    totals = {
        "terminated": 0,
        "truncated": 0,
        "scores": 0,
        "misses": 0,
        "hit_episodes": 0,
        "agent_hits": 0,
        "opponent_hits": 0,
        "rally": 0,
        "legal_landings": 0,
        "reward": 0.0,
        "steps": 0,
        "tracking_error": 0.0,
        "joint_speed": 0.0,
        "robot_enabled": 0,
    }
    point_reasons: dict[str, int] = {}

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        tracking_samples = 0
        tracking_total = 0.0
        joint_speed_total = 0.0
        info = {}
        terminated = False
        truncated = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            tracking_total += float(info.get("agent_tracking_error", 0.0))
            joint_speed_total += sum(abs(value) for value in info.get("agent_joint_velocities", [0.0, 0.0, 0.0]))
            tracking_samples += 1
            done = terminated or truncated

        totals["terminated"] += int(terminated)
        totals["truncated"] += int(truncated)
        totals["scores"] += int(info["agent_score"])
        totals["misses"] += int(info["agent_miss"])
        totals["hit_episodes"] += int(info["agent_hits"] > 0)
        totals["agent_hits"] += int(info["agent_hits"])
        totals["opponent_hits"] += int(info["opponent_hits"])
        totals["rally"] += int(info["rally_length"])
        totals["legal_landings"] += int(info["legal_landings"])
        totals["reward"] += episode_reward
        totals["steps"] += int(info["steps"])
        totals["tracking_error"] += tracking_total / max(tracking_samples, 1)
        totals["joint_speed"] += joint_speed_total / max(tracking_samples, 1)
        totals["robot_enabled"] += int(info.get("robot_arm_enabled", False))
        reason = str(info.get("point_reason", "unknown"))
        point_reasons[reason] = point_reasons.get(reason, 0) + 1

    env.close()
    return {
        "stage": 14,
        "episodes": episodes,
        "robot_arm_enabled_rate": totals["robot_enabled"] / episodes,
        "normal_end_rate": totals["terminated"] / episodes,
        "truncated_rate": totals["truncated"] / episodes,
        "win_rate": totals["scores"] / episodes,
        "miss_rate": totals["misses"] / episodes,
        "hit_rate": totals["hit_episodes"] / episodes,
        "avg_agent_hits": totals["agent_hits"] / episodes,
        "avg_opponent_hits": totals["opponent_hits"] / episodes,
        "avg_rally_length": totals["rally"] / episodes,
        "avg_legal_landings": totals["legal_landings"] / episodes,
        "avg_tracking_error": totals["tracking_error"] / episodes,
        "avg_joint_speed": totals["joint_speed"] / episodes,
        "point_reasons": point_reasons,
        "avg_reward": totals["reward"] / episodes,
        "avg_steps": totals["steps"] / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Stage 14 robot-arm ping pong policy.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_robot_arm_stage14")
    parser.add_argument("--episodes", type=int, default=100, help="Use 0 with --render to watch until the window is closed.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.render and args.episodes == 0:
        env = RobotArmPingPongEnv(render_mode="human")
        model = PPO.load(args.model_path)
        obs, _ = env.reset(seed=args.seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

    metrics = evaluate_robot_arm_model(args.model_path, args.episodes, args.render, args.seed)
    write_metrics(metrics, args.json_path)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
