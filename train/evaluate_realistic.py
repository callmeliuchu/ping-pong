from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from pingpong_rl.envs import RealisticPingPongEnv
from train.eval_utils import ROOT, print_metrics, write_metrics


def evaluate_realistic_model(
    model_path: Path,
    episodes: int,
    render: bool = False,
    seed: int = 0,
) -> dict:
    env = RealisticPingPongEnv(render_mode="human" if render else None)
    model = PPO.load(model_path)

    terminated_count = 0
    truncated_count = 0
    total_scores = 0
    total_misses = 0
    episodes_with_hit = 0
    total_agent_hits = 0
    total_opponent_hits = 0
    total_rally = 0
    total_legal_landings = 0
    total_reward = 0.0
    total_steps = 0
    total_spin = 0.0
    total_contact_quality = 0.0
    rally_8_count = 0
    rally_10_count = 0
    clean_success_count = 0
    point_reasons: dict[str, int] = {}

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        episode_spin_samples = 0
        episode_spin_total = 0.0
        info = {}
        terminated = False
        truncated = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_spin_total += abs(env.ball_spin)
            episode_spin_samples += 1
            done = terminated or truncated

        rally_length = int(info["rally_length"])
        terminated_count += int(terminated)
        truncated_count += int(truncated)
        total_scores += int(info["agent_score"])
        total_misses += int(info["agent_miss"])
        episodes_with_hit += int(info["agent_hits"] > 0)
        total_agent_hits += int(info["agent_hits"])
        total_opponent_hits += int(info["opponent_hits"])
        total_rally += rally_length
        total_legal_landings += int(info["legal_landings"])
        total_reward += episode_reward
        total_steps += int(info["steps"])
        total_spin += episode_spin_total / max(episode_spin_samples, 1)
        total_contact_quality += float(info["contact_quality"])
        rally_8_count += int(rally_length >= 8)
        rally_10_count += int(rally_length >= 10)
        clean_success_count += int(info.get("rally_success", False) and not info["agent_score"] and not info["agent_miss"])
        reason = str(info.get("point_reason", "unknown"))
        point_reasons[reason] = point_reasons.get(reason, 0) + 1

    env.close()
    return {
        "stage": 7,
        "episodes": episodes,
        "normal_end_rate": terminated_count / episodes,
        "truncated_rate": truncated_count / episodes,
        "win_rate": total_scores / episodes,
        "miss_rate": total_misses / episodes,
        "hit_rate": episodes_with_hit / episodes,
        "avg_agent_hits": total_agent_hits / episodes,
        "avg_opponent_hits": total_opponent_hits / episodes,
        "avg_rally_length": total_rally / episodes,
        "rally_8_rate": rally_8_count / episodes,
        "rally_10_rate": rally_10_count / episodes,
        "rule_clean_success_rate": clean_success_count / episodes,
        "avg_legal_landings": total_legal_landings / episodes,
        "avg_abs_spin": total_spin / episodes,
        "avg_contact_quality": total_contact_quality / episodes,
        "point_reasons": point_reasons,
        "avg_reward": total_reward / episodes,
        "avg_steps": total_steps / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Stage 7 realistic paddle policy.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_realistic_stage7")
    parser.add_argument("--episodes", type=int, default=100, help="Use 0 with --render to watch until the window is closed.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.render and args.episodes == 0:
        env = RealisticPingPongEnv(render_mode="human")
        model = PPO.load(args.model_path)
        obs, _ = env.reset(seed=args.seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

    metrics = evaluate_realistic_model(args.model_path, args.episodes, args.render, args.seed)
    write_metrics(metrics, args.json_path)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
