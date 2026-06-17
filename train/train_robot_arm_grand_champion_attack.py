from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pingpong_rl.envs.robot_arm_grand_champion_attack_env import (
    RobotArmGrandChampionAttackConfig,
    RobotArmGrandChampionAttackEnv,
)
from train.eval_utils import ROOT, write_metrics
from train.evaluate_robot_arm_grand_champion_attack import (
    default_grand_champion_attack_opponent_paths,
    evaluate_robot_arm_grand_champion_attack_model,
)


def make_env(opponent_model_paths: list[Path]):
    def _init():
        config = RobotArmGrandChampionAttackConfig(opponent_model_paths=tuple(str(path) for path in opponent_model_paths))
        return Monitor(RobotArmGrandChampionAttackEnv(render_mode=None, config=config))

    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 21 grand champion robot-arm league with PPO.")
    parser.add_argument("--timesteps", type=int, default=120_000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation", type=int, default=1)
    parser.add_argument("--base-model-path", type=Path, default=ROOT / "models" / "passed" / "ppo_stage20")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "ppo_robot_arm_grand_champion_attack_stage21")
    parser.add_argument("--opponent-model-paths", type=Path, nargs="*", default=None)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--eval-json-path", type=Path, default=None)
    args = parser.parse_args()

    opponent_paths = (
        args.opponent_model_paths
        if args.opponent_model_paths is not None
        else default_grand_champion_attack_opponent_paths()
    )
    if not opponent_paths:
        raise ValueError("Stage 21 training needs at least one robot-arm opponent model.")

    check_env(RobotArmGrandChampionAttackEnv(config=RobotArmGrandChampionAttackConfig()))

    vec_env = DummyVecEnv([make_env(opponent_paths) for _ in range(args.num_envs)])
    vec_env.seed(args.seed)
    eval_env = Monitor(
        RobotArmGrandChampionAttackEnv(
            config=RobotArmGrandChampionAttackConfig(opponent_model_paths=tuple(str(path) for path in opponent_paths))
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.timesteps // 4 // args.num_envs, 1),
        save_path=str(ROOT / "models" / "checkpoints"),
        name_prefix=f"ppo_robot_arm_grand_champion_attack_stage21_gen{args.generation}",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ROOT / "models" / "best" / "stage21"),
        log_path=str(ROOT / "logs" / "ppo_robot_arm_grand_champion_attack_stage21" / "eval"),
        eval_freq=max(args.timesteps // 8 // args.num_envs, 1),
        n_eval_episodes=min(args.eval_episodes, 50),
        deterministic=True,
        render=False,
    )

    model = PPO.load(args.base_model_path, env=vec_env)
    model.verbose = 1
    model.learn(total_timesteps=args.timesteps, callback=CallbackList([checkpoint_callback, eval_callback]))
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    generation_path = ROOT / "models" / "selfplay" / "stage21" / f"gen_{args.generation}.zip"
    generation_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(generation_path)
    vec_env.close()
    eval_env.close()

    metrics = evaluate_robot_arm_grand_champion_attack_model(args.model_path, opponent_paths, args.eval_episodes, seed=args.seed)
    write_metrics(
        metrics,
        args.eval_json_path
        or ROOT / "logs" / "ppo_robot_arm_grand_champion_attack_stage21" / f"gen_{args.generation}_eval.json",
    )


if __name__ == "__main__":
    main()
