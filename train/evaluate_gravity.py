from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from pingpong_rl.envs import GravityPingPongEnv
from train.eval_utils import ROOT, print_metrics, write_metrics


def evaluate_gravity_model(
    model_path: Path,
    episodes: int,
    render: bool = False,
    seed: int = 0,
) -> dict:
    env = GravityPingPongEnv(render_mode="human" if render else None)
    model = PPO.load(model_path)

    terminated_count = 0
    truncated_count = 0
    total_scores = 0
    total_misses = 0
    episodes_with_hit = 0
    total_agent_hits = 0
    total_opponent_hits = 0
    total_rally = 0
    total_reward = 0.0
    total_steps = 0
    rally_10_count = 0
    rally_15_count = 0
    long_rally_success_count = 0
    max_rally_length = 0
    total_legal_landings = 0
    rule_clean_success_count = 0
    point_reasons: dict[str, int] = {}

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        info = {}
        terminated = False
        truncated = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            total_legal_landings += int(info.get("legal_landing", False))
            done = terminated or truncated

        terminated_count += int(terminated)
        truncated_count += int(truncated)
        total_scores += int(info["agent_score"])
        total_misses += int(info["agent_miss"])
        long_rally_success_count += int(info.get("rally_success", False))
        rule_clean_success_count += int(
            info.get("rules_enabled", False)
            and info.get("rally_success", False)
            and not info.get("agent_score", False)
            and not info.get("agent_miss", False)
        )
        reason = str(info.get("point_reason", "unknown"))
        point_reasons[reason] = point_reasons.get(reason, 0) + 1
        episodes_with_hit += int(info["agent_hits"] > 0)
        total_agent_hits += int(info["agent_hits"])
        total_opponent_hits += int(info["opponent_hits"])
        rally_length = int(info["rally_length"])
        total_rally += rally_length
        rally_10_count += int(rally_length >= 10)
        rally_15_count += int(rally_length >= 15)
        max_rally_length = max(max_rally_length, rally_length)
        total_reward += episode_reward
        total_steps += int(info["steps"])

    env.close()
    return {
        "stage": 6,
        "episodes": episodes,
        "normal_end_rate": terminated_count / episodes,
        "truncated_rate": truncated_count / episodes,
        "win_rate": total_scores / episodes,
        "miss_rate": total_misses / episodes,
        "hit_rate": episodes_with_hit / episodes,
        "episode_hit_rate": episodes_with_hit / episodes,
        "avg_agent_hits": total_agent_hits / episodes,
        "avg_opponent_hits": total_opponent_hits / episodes,
        "avg_rally_length": total_rally / episodes,
        "max_rally_length": max_rally_length,
        "rally_10_rate": rally_10_count / episodes,
        "rally_15_rate": rally_15_count / episodes,
        "long_rally_success_rate": long_rally_success_count / episodes,
        "rule_clean_success_rate": rule_clean_success_count / episodes,
        "avg_legal_landings": total_legal_landings / episodes,
        "point_reasons": point_reasons,
        "avg_reward": total_reward / episodes,
        "avg_steps": total_steps / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained Stage 6 gravity PPO policy.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_gravity_stage6")
    parser.add_argument("--episodes", type=int, default=100, help="Use 0 with --render to watch until the window is closed.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.render and args.episodes == 0:
        env = GravityPingPongEnv(render_mode="human")
        model = PPO.load(args.model_path)
        obs, _ = env.reset(seed=args.seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                obs, _ = env.reset()

    metrics = evaluate_gravity_model(args.model_path, args.episodes, args.render, args.seed)
    write_metrics(metrics, args.json_path)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
