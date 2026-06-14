from __future__ import annotations

from pingpong_rl.envs import GravityPingPongEnv


def main() -> None:
    env = GravityPingPongEnv(render_mode="human")
    obs, _ = env.reset()

    while True:
        action = 0
        if env.ball_vx < 0:
            target_y = env.predict_ball_y_at_x(env.config.paddle_x)
        else:
            target_y = env.config.table_y - 90
        if target_y < env.agent_y - 5:
            action = 1
        elif target_y > env.agent_y + 5:
            action = 2

        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
