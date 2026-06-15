from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_baselines3.common.env_checker import check_env

from pingpong_rl.envs import CatchEnv, CompetitiveRealisticEnv, GravityPingPongEnv, PongEnv, RealisticPingPongEnv, make_pong_config
from train.eval_utils import ROOT, evaluate_pong, evaluate_stage1, smoke_gravity, write_metrics
from train.evaluate_gravity import evaluate_gravity_model
from train.evaluate_realistic import evaluate_realistic_model
from train.evaluate_competitive import evaluate_competitive_model


@dataclass(frozen=True)
class StageResult:
    stage: int
    passed: bool
    metrics: dict[str, Any]
    reason: str
    suggestion: str
    model_path: Path | None = None


def _zip_path(path: Path) -> Path:
    return path if path.suffix == ".zip" else path.with_suffix(".zip")


def _copy_passed_model(stage: int, model_path: Path, passed_dir: Path) -> None:
    source = _zip_path(model_path)
    if not source.exists():
        return
    passed_dir.mkdir(parents=True, exist_ok=True)
    destination = passed_dir / f"ppo_stage{stage}.zip"
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def _preferred_model_path(stage: int, root_model_path: Path) -> Path:
    passed_model_path = ROOT / "models" / "passed" / f"ppo_stage{stage}"
    return passed_model_path if _zip_path(passed_model_path).exists() else root_model_path


def _check_envs() -> None:
    check_env(CatchEnv())
    for stage in (2, 3, 4):
        check_env(PongEnv(config=make_pong_config(stage)))
    check_env(PongEnv(config=make_pong_config(5, str(ROOT / "models" / "ppo_pong_stage4"))))
    check_env(GravityPingPongEnv())
    check_env(RealisticPingPongEnv())
    check_env(CompetitiveRealisticEnv())


def _stage_pass(stage: int, metrics: dict[str, Any]) -> tuple[bool, str]:
    if stage == 1:
        return metrics["hit_rate"] >= 0.90, "hit_rate >= 0.90"
    if stage == 2:
        return (
            metrics["win_rate"] >= 0.70 and metrics["episode_hit_rate"] >= 0.80,
            "win_rate >= 0.70 and episode_hit_rate >= 0.80",
        )
    if stage == 3:
        return (
            metrics["episode_hit_rate"] >= 0.85
            and metrics["avg_rally_length"] >= 5.0
            and metrics["win_rate"] >= 0.25,
            "episode_hit_rate >= 0.85, avg_rally_length >= 5.0, win_rate >= 0.25",
        )
    if stage == 4:
        return (
            metrics["episode_hit_rate"] >= 0.85
            and metrics["avg_rally_length"] >= 8.0
            and metrics["avg_reward"] > 0.0,
            "episode_hit_rate >= 0.85, avg_rally_length >= 8.0, avg_reward > 0",
        )
    if stage == 5:
        return (
            0.45 <= metrics["win_rate"] <= 0.65 and metrics["avg_reward"] >= 0.0,
            "0.45 <= win_rate <= 0.65 and avg_reward >= 0",
        )
    if stage == 6:
        if "hit_rate" in metrics:
            return (
                metrics["normal_end_rate"] >= 0.80
                and metrics["rally_10_rate"] >= 0.90
                and metrics["rally_15_rate"] >= 0.80
                and metrics["long_rally_success_rate"] >= 0.80
                and metrics.get("rule_clean_success_rate", 0.0) >= 0.80
                and metrics.get("avg_legal_landings", 0.0) >= 10.0
                and metrics["avg_reward"] > 0.0,
                "normal_end_rate >= 0.80, rally_10_rate >= 0.90, rally_15_rate >= 0.80, long_rally_success_rate >= 0.80, rule_clean_success_rate >= 0.80, avg_legal_landings >= 10, avg_reward > 0",
            )
        return metrics["normal_end_rate"] >= 0.80, "normal_end_rate >= 0.80"
    if stage == 7:
        return (
            metrics["normal_end_rate"] >= 0.80
            and metrics["rally_10_rate"] >= 0.70
            and metrics["rule_clean_success_rate"] >= 0.70
            and metrics["avg_abs_spin"] >= 0.10
            and metrics["avg_contact_quality"] >= 0.20
            and metrics["avg_reward"] > 0.0,
            "normal_end_rate >= 0.80, rally_10_rate >= 0.70, rule_clean_success_rate >= 0.70, avg_abs_spin >= 0.10, avg_contact_quality >= 0.20, avg_reward > 0",
        )
    if stage == 8:
        return (
            metrics["normal_end_rate"] >= 0.85
            and metrics["win_rate"] >= 0.60
            and metrics["avg_rally_length"] >= 6.0
            and metrics["attack_attempt_rate"] >= 0.05
            and metrics["attack_success_rate"] >= 0.25
            and metrics["attack_landing_rate"] >= 0.20
            and metrics["avg_abs_spin"] >= 0.25
            and metrics["avg_paddle_x_range"] >= 35.0
            and metrics["avg_reward"] > 0.0,
            "normal_end_rate >= 0.85, win_rate >= 0.60, avg_rally_length >= 6.0, attack_attempt_rate >= 0.05, attack_success_rate >= 0.25, attack_landing_rate >= 0.20, avg_abs_spin >= 0.25, avg_paddle_x_range >= 35, avg_reward > 0",
        )
    raise ValueError(stage)


def _suggestion(stage: int) -> str:
    suggestions = {
        1: "python -m train.train_stage1 --timesteps 100000 --model-path models/ppo_catch_stage1",
        2: "python -m train.train_pong --stage 2 --load-model-path models/ppo_pong_stage2 --timesteps 500000",
        3: "python -m train.train_pong --stage 3 --load-model-path models/ppo_pong_stage2 --timesteps 500000",
        4: "python -m train.train_pong --stage 4 --load-model-path models/ppo_pong_stage3 --timesteps 500000",
        5: "python -m train.self_play_stage5 --base-model-path models/ppo_pong_stage4 --timesteps 200000",
        6: "python -m train.train_gravity --load-model-path models/ppo_gravity_stage6 --timesteps 500000",
        7: "python -m train.train_realistic --timesteps 500000",
        8: "python -m train.train_competitive --timesteps 500000",
    }
    return suggestions[stage]


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _print_table(results: list[StageResult]) -> None:
    print("stage | status | key metrics | gate")
    print("--- | --- | --- | ---")
    for result in results:
        metrics = result.metrics
        if result.stage == 1:
            key_metrics = f"hit={metrics['hit_rate']:.3f}, reward={metrics['avg_reward']:.3f}"
        elif result.stage in (2, 3, 4, 5):
            key_metrics = (
                f"win={metrics['win_rate']:.3f}, hit={metrics['episode_hit_rate']:.3f}, "
                f"rally={metrics['avg_rally_length']:.2f}, reward={metrics['avg_reward']:.3f}"
            )
        elif result.stage == 6:
            if "hit_rate" in metrics:
                key_metrics = (
                    f"hit={metrics['hit_rate']:.3f}, rally={metrics['avg_rally_length']:.2f}, "
                    f"r10={metrics['rally_10_rate']:.3f}, r15={metrics['rally_15_rate']:.3f}, "
                    f"rule_clean={metrics.get('rule_clean_success_rate', 0.0):.3f}, "
                    f"legal_land={metrics.get('avg_legal_landings', 0.0):.1f}, reward={metrics['avg_reward']:.3f}"
                )
            else:
                key_metrics = f"normal_end={metrics['normal_end_rate']:.3f}, avg_steps={metrics['avg_steps']:.1f}"
        elif result.stage == 7:
            key_metrics = (
                f"hit={metrics['hit_rate']:.3f}, rally={metrics['avg_rally_length']:.2f}, "
                f"r10={metrics['rally_10_rate']:.3f}, clean={metrics['rule_clean_success_rate']:.3f}, "
                f"spin={metrics['avg_abs_spin']:.2f}, contact={metrics['avg_contact_quality']:.2f}, "
                f"reward={metrics['avg_reward']:.3f}"
            )
        else:
            key_metrics = (
                f"win={metrics['win_rate']:.3f}, rally={metrics['avg_rally_length']:.2f}, "
                f"attempt={metrics['attack_attempt_rate']:.3f}, attack={metrics['attack_success_rate']:.3f}, "
                f"land={metrics['attack_landing_rate']:.3f}, spin={metrics['avg_abs_spin']:.2f}, "
                f"x_range={metrics['avg_paddle_x_range']:.1f}, reward={metrics['avg_reward']:.3f}"
            )
        print(f"{result.stage} | {'PASS' if result.passed else 'FAIL'} | {key_metrics} | {result.reason}")

    failures = [result for result in results if not result.passed]
    if failures:
        print("\nfailed stages:")
        for failure in failures:
            print(f"stage {failure.stage}: {failure.suggestion}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all eight ping pong RL stages.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--stage6-episodes", type=int, default=100)
    parser.add_argument("--stage7-episodes", type=int, default=100)
    parser.add_argument("--stage8-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-check-env", action="store_true")
    parser.add_argument("--json-dir", type=Path, default=ROOT / "logs" / "validation")
    parser.add_argument("--passed-dir", type=Path, default=ROOT / "models" / "passed")
    parser.add_argument("--no-copy-passed", action="store_true")
    args = parser.parse_args()

    if not args.skip_check_env:
        _check_envs()

    stage_models = {
        1: _preferred_model_path(1, ROOT / "models" / "ppo_catch_stage1"),
        2: _preferred_model_path(2, ROOT / "models" / "ppo_pong_stage2"),
        3: _preferred_model_path(3, ROOT / "models" / "ppo_pong_stage3"),
        4: _preferred_model_path(4, ROOT / "models" / "ppo_pong_stage4"),
        5: _preferred_model_path(5, ROOT / "models" / "ppo_pong_stage5"),
        6: _preferred_model_path(6, ROOT / "models" / "ppo_gravity_stage6"),
        7: _preferred_model_path(7, ROOT / "models" / "ppo_realistic_stage7"),
        8: _preferred_model_path(8, ROOT / "models" / "ppo_competitive_stage8"),
    }

    raw_results: list[tuple[int, dict[str, Any], Path | None]] = [
        (1, evaluate_stage1(stage_models[1], args.episodes, seed=args.seed), stage_models[1]),
        (2, evaluate_pong(2, stage_models[2], episodes=args.episodes, seed=args.seed), stage_models[2]),
        (3, evaluate_pong(3, stage_models[3], episodes=args.episodes, seed=args.seed), stage_models[3]),
        (4, evaluate_pong(4, stage_models[4], episodes=args.episodes, seed=args.seed), stage_models[4]),
        (
            5,
            evaluate_pong(
                5,
                stage_models[5],
                opponent_model_path=str(stage_models[4]),
                episodes=args.episodes,
                seed=args.seed,
            ),
            stage_models[5],
        ),
        (6, evaluate_gravity_model(stage_models[6], args.stage6_episodes, seed=args.seed), stage_models[6]),
        (7, evaluate_realistic_model(stage_models[7], args.stage7_episodes, seed=args.seed), stage_models[7]),
        (8, evaluate_competitive_model(stage_models[8], args.stage8_episodes, seed=args.seed), stage_models[8]),
    ]

    results: list[StageResult] = []
    for stage, metrics, model_path in raw_results:
        passed, reason = _stage_pass(stage, metrics)
        write_metrics(metrics, args.json_dir / f"stage{stage}.json")
        if passed and model_path is not None and not args.no_copy_passed:
            _copy_passed_model(stage, model_path, args.passed_dir)
        results.append(
            StageResult(
                stage=stage,
                passed=passed,
                metrics=metrics,
                reason=reason,
                suggestion=_suggestion(stage),
                model_path=model_path,
            )
        )

    _print_table(results)
    if not all(result.passed for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
