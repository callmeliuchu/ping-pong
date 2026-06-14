from __future__ import annotations

from pingpong_rl.envs import RealisticPingPongEnv


def main() -> None:
    env = RealisticPingPongEnv(render_mode="human")
    env.reset()
    while True:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()


if __name__ == "__main__":
    main()
