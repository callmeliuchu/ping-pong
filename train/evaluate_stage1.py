from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from pingpong_rl.envs import CatchEnv


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Stage 1 catch policy.")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_catch_stage1")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    env = CatchEnv(render_mode="human" if args.render else None)
    model = PPO.load(args.model_path)

    hits = 0
    misses = 0
    total_reward = 0.0
    total_steps = 0

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=episode)
        done = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            done = terminated or truncated

        hits += int(info["agent_hit"])
        misses += int(info["agent_miss"])
        total_reward += episode_reward
        total_steps += int(info["steps"])

    env.close()

    print(f"episodes={args.episodes}")
    print(f"hit_rate={hits / args.episodes:.3f}")
    print(f"miss_rate={misses / args.episodes:.3f}")
    print(f"avg_reward={total_reward / args.episodes:.3f}")
    print(f"avg_steps={total_steps / args.episodes:.1f}")


if __name__ == "__main__":
    main()

