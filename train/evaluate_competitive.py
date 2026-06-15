from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from pingpong_rl.envs import CompetitiveRealisticEnv
from train.eval_utils import ROOT, print_metrics, write_metrics


def evaluate_competitive_model(
    model_path: Path,
    episodes: int,
    render: bool = False,
    seed: int = 0,
) -> dict:
    env = CompetitiveRealisticEnv(render_mode="human" if render else None)
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
    total_paddle_x_range = 0.0
    attack_attempt_episodes = 0
    attack_landing_episodes = 0
    attack_score_episodes = 0
    attack_success_count = 0
    point_reasons: dict[str, int] = {}

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        episode_spin_total = 0.0
        episode_spin_samples = 0
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

        terminated_count += int(terminated)
        truncated_count += int(truncated)
        total_scores += int(info["agent_score"])
        total_misses += int(info["agent_miss"])
        episodes_with_hit += int(info["agent_hits"] > 0)
        total_agent_hits += int(info["agent_hits"])
        total_opponent_hits += int(info["opponent_hits"])
        total_rally += int(info["rally_length"])
        total_legal_landings += int(info["legal_landings"])
        total_reward += episode_reward
        total_steps += int(info["steps"])
        total_spin += episode_spin_total / max(episode_spin_samples, 1)
        total_contact_quality += float(info["contact_quality"])
        total_paddle_x_range += float(info["paddle_x_range"])
        attack_attempt_episodes += int(info["attack_attempts"] > 0)
        attack_landing_episodes += int(info["attack_landings"] > 0)
        attack_score_episodes += int(info["attack_scores"] > 0)
        attack_success_count += int(info["attack_success"])
        reason = str(info.get("point_reason", "unknown"))
        point_reasons[reason] = point_reasons.get(reason, 0) + 1

    env.close()
    return {
        "stage": 8,
        "episodes": episodes,
        "normal_end_rate": terminated_count / episodes,
        "truncated_rate": truncated_count / episodes,
        "win_rate": total_scores / episodes,
        "miss_rate": total_misses / episodes,
        "hit_rate": episodes_with_hit / episodes,
        "avg_agent_hits": total_agent_hits / episodes,
        "avg_opponent_hits": total_opponent_hits / episodes,
        "avg_rally_length": total_rally / episodes,
        "avg_legal_landings": total_legal_landings / episodes,
        "attack_attempt_rate": attack_attempt_episodes / episodes,
        "attack_success_rate": attack_success_count / episodes,
        "avg_abs_spin": total_spin / episodes,
        "avg_contact_quality": total_contact_quality / episodes,
        "avg_paddle_x_range": total_paddle_x_range / episodes,
        "point_reasons": point_reasons,
        "attack_landing_rate": attack_landing_episodes / episodes,
        "attack_score_rate": attack_score_episodes / episodes,
        "avg_reward": total_reward / episodes,
        "avg_steps": total_steps / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Stage 8 competitive realistic policy.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_competitive_stage8")
    parser.add_argument("--episodes", type=int, default=100, help="Use 0 with --render to watch until the window is closed.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.render and args.episodes == 0:
        env = CompetitiveRealisticEnv(render_mode="human")
        model = PPO.load(args.model_path)
        obs, _ = env.reset(seed=args.seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

    metrics = evaluate_competitive_model(args.model_path, args.episodes, args.render, args.seed)
    write_metrics(metrics, args.json_path)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
