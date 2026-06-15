from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from pingpong_rl.envs import VarietyTechniqueEnv
from train.eval_utils import ROOT, print_metrics, write_metrics


def evaluate_variety_model(
    model_path: Path,
    episodes: int,
    render: bool = False,
    seed: int = 0,
) -> dict:
    env = VarietyTechniqueEnv(render_mode="human" if render else None)
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
        "spin": 0.0,
        "paddle_x_range": 0.0,
        "loop_landing_episodes": 0,
        "drive_landing_episodes": 0,
        "smash_landing_episodes": 0,
        "chop_landing_episodes": 0,
        "high_arc_episodes": 0,
        "deep_drive_episodes": 0,
        "style_match_landings": 0,
        "unique_styles": 0,
        "max_topspin": 0.0,
        "max_backspin": 0.0,
        "angled_hit_rate": 0.0,
        "block_rate": 0.0,
    }
    point_reasons: dict[str, int] = {}

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        spin_total = 0.0
        spin_samples = 0
        info = {}
        terminated = False
        truncated = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            spin_total += abs(env.ball_spin)
            spin_samples += 1
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
        totals["spin"] += spin_total / max(spin_samples, 1)
        totals["paddle_x_range"] += float(info["paddle_x_range"])
        totals["loop_landing_episodes"] += int(info["loop_landings"] > 0)
        totals["drive_landing_episodes"] += int(info["drive_landings"] > 0)
        totals["smash_landing_episodes"] += int(info["smash_landings"] > 0)
        totals["chop_landing_episodes"] += int(info["chop_landings"] > 0)
        totals["high_arc_episodes"] += int(info["high_arc_landings"] > 0)
        totals["deep_drive_episodes"] += int(info["deep_drive_landings"] > 0)
        totals["style_match_landings"] += int(info["style_match_landings"])
        totals["unique_styles"] += int(info["unique_styles_landed"])
        totals["max_topspin"] += float(info["max_topspin"])
        totals["max_backspin"] += float(info["max_backspin"])
        totals["angled_hit_rate"] += float(info["angled_hit_rate"])
        totals["block_rate"] += float(info["block_rate"])
        reason = str(info.get("point_reason", "unknown"))
        point_reasons[reason] = point_reasons.get(reason, 0) + 1

    env.close()
    return {
        "stage": 11,
        "episodes": episodes,
        "normal_end_rate": totals["terminated"] / episodes,
        "truncated_rate": totals["truncated"] / episodes,
        "win_rate": totals["scores"] / episodes,
        "miss_rate": totals["misses"] / episodes,
        "hit_rate": totals["hit_episodes"] / episodes,
        "avg_agent_hits": totals["agent_hits"] / episodes,
        "avg_opponent_hits": totals["opponent_hits"] / episodes,
        "avg_rally_length": totals["rally"] / episodes,
        "avg_legal_landings": totals["legal_landings"] / episodes,
        "avg_abs_spin": totals["spin"] / episodes,
        "avg_paddle_x_range": totals["paddle_x_range"] / episodes,
        "loop_landing_rate": totals["loop_landing_episodes"] / episodes,
        "drive_landing_rate": totals["drive_landing_episodes"] / episodes,
        "smash_landing_rate": totals["smash_landing_episodes"] / episodes,
        "chop_landing_rate": totals["chop_landing_episodes"] / episodes,
        "high_arc_rate": totals["high_arc_episodes"] / episodes,
        "deep_drive_rate": totals["deep_drive_episodes"] / episodes,
        "avg_style_match_landings": totals["style_match_landings"] / episodes,
        "avg_unique_styles_landed": totals["unique_styles"] / episodes,
        "avg_max_topspin": totals["max_topspin"] / episodes,
        "avg_max_backspin": totals["max_backspin"] / episodes,
        "avg_angled_hit_rate": totals["angled_hit_rate"] / episodes,
        "avg_block_rate": totals["block_rate"] / episodes,
        "point_reasons": point_reasons,
        "avg_reward": totals["reward"] / episodes,
        "avg_steps": totals["steps"] / episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Stage 11 varied-stroke policy.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_variety_stage11")
    parser.add_argument("--episodes", type=int, default=100, help="Use 0 with --render to watch until the window is closed.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.render and args.episodes == 0:
        env = VarietyTechniqueEnv(render_mode="human")
        model = PPO.load(args.model_path)
        obs, _ = env.reset(seed=args.seed)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()

    metrics = evaluate_variety_model(args.model_path, args.episodes, args.render, args.seed)
    write_metrics(metrics, args.json_path)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
