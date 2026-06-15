from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from pingpong_rl.envs.self_play_variety_env import SelfPlayVarietyConfig, SelfPlayVarietyEnv
from train.eval_utils import ROOT, print_metrics, write_metrics


def evaluate_selfplay_model(
    model_path: Path,
    opponent_model_path: Path,
    episodes: int,
    render: bool = False,
    seed: int = 0,
) -> dict:
    config = SelfPlayVarietyConfig(opponent_model_path=str(opponent_model_path))
    env = SelfPlayVarietyEnv(render_mode="human" if render else None, config=config)
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
        "reward": 0.0,
        "steps": 0,
        "loop_landing_episodes": 0,
        "drive_landing_episodes": 0,
        "smash_landing_episodes": 0,
        "high_arc_episodes": 0,
        "unique_styles": 0,
        "style_match_landings": 0,
        "opponent_loaded": 0,
    }
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
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        totals["terminated"] += int(terminated)
        totals["truncated"] += int(truncated)
        totals["scores"] += int(info["agent_score"])
        totals["misses"] += int(info["agent_miss"])
        totals["hit_episodes"] += int(info["agent_hits"] > 0)
        totals["agent_hits"] += int(info["agent_hits"])
        totals["opponent_hits"] += int(info["opponent_hits"])
        totals["rally"] += int(info["rally_length"])
        totals["reward"] += episode_reward
        totals["steps"] += int(info["steps"])
        totals["loop_landing_episodes"] += int(info["loop_landings"] > 0)
        totals["drive_landing_episodes"] += int(info["drive_landings"] > 0)
        totals["smash_landing_episodes"] += int(info["smash_landings"] > 0)
        totals["high_arc_episodes"] += int(info["high_arc_landings"] > 0)
        totals["unique_styles"] += int(info["unique_styles_landed"])
        totals["style_match_landings"] += int(info["style_match_landings"])
        totals["opponent_loaded"] += int(info["opponent_model_loaded"])
        reason = str(info.get("point_reason", "unknown"))
        point_reasons[reason] = point_reasons.get(reason, 0) + 1

    env.close()
    return {
        "stage": 12,
        "episodes": episodes,
        "normal_end_rate": totals["terminated"] / episodes,
        "truncated_rate": totals["truncated"] / episodes,
        "win_rate": totals["scores"] / episodes,
        "miss_rate": totals["misses"] / episodes,
        "hit_rate": totals["hit_episodes"] / episodes,
        "avg_agent_hits": totals["agent_hits"] / episodes,
        "avg_opponent_hits": totals["opponent_hits"] / episodes,
        "avg_rally_length": totals["rally"] / episodes,
        "loop_landing_rate": totals["loop_landing_episodes"] / episodes,
        "drive_landing_rate": totals["drive_landing_episodes"] / episodes,
        "smash_landing_rate": totals["smash_landing_episodes"] / episodes,
        "high_arc_rate": totals["high_arc_episodes"] / episodes,
        "avg_unique_styles_landed": totals["unique_styles"] / episodes,
        "avg_style_match_landings": totals["style_match_landings"] / episodes,
        "opponent_loaded_rate": totals["opponent_loaded"] / episodes,
        "point_reasons": point_reasons,
        "avg_reward": totals["reward"] / episodes,
        "avg_steps": totals["steps"] / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Stage 12 self-play policy against a historical model.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_selfplay_stage12")
    parser.add_argument("--opponent-model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage11")
    parser.add_argument("--episodes", type=int, default=100, help="Use 0 with --render to watch until the window is closed.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.render and args.episodes == 0:
        config = SelfPlayVarietyConfig(opponent_model_path=str(args.opponent_model_path))
        env = SelfPlayVarietyEnv(render_mode="human", config=config)
        model = PPO.load(args.model_path)
        obs, _ = env.reset(seed=args.seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

    metrics = evaluate_selfplay_model(args.model_path, args.opponent_model_path, args.episodes, args.render, args.seed)
    write_metrics(metrics, args.json_path)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
