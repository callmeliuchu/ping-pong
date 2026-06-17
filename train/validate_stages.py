from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stable_baselines3.common.env_checker import check_env

from pingpong_rl.envs import (
    AdvancedStrokesEnv,
    CatchEnv,
    CompactTechniqueEnv,
    CompetitiveRealisticEnv,
    GravityPingPongEnv,
    LeagueSelfPlayEnv,
    PongEnv,
    RealisticPingPongEnv,
    RobotArmAttackLeagueEnv,
    RobotArmLeagueEnv,
    RobotArmPingPongEnv,
    RobotArmTacticalLeagueEnv,
    SelfPlayVarietyEnv,
    VarietyTechniqueEnv,
    make_pong_config,
)
from pingpong_rl.envs.league_self_play_env import LeagueSelfPlayConfig
from pingpong_rl.envs.robot_arm_attack_league_env import RobotArmAttackLeagueConfig
from pingpong_rl.envs.robot_arm_league_env import RobotArmLeagueConfig
from pingpong_rl.envs.robot_arm_tactical_league_env import RobotArmTacticalLeagueConfig
from pingpong_rl.envs.self_play_variety_env import SelfPlayVarietyConfig
from train.eval_utils import ROOT, evaluate_pong, evaluate_stage1, smoke_gravity, write_metrics
from train.evaluate_advanced import evaluate_advanced_model
from train.evaluate_compact import evaluate_compact_model
from train.evaluate_gravity import evaluate_gravity_model
from train.evaluate_league_stage13 import default_opponent_paths, evaluate_league_model
from train.evaluate_realistic import evaluate_realistic_model
from train.evaluate_competitive import evaluate_competitive_model
from train.evaluate_robot_arm import evaluate_robot_arm_model
from train.evaluate_robot_arm_attack import default_attack_opponent_paths, evaluate_robot_arm_attack_model
from train.evaluate_robot_arm_league import default_robot_arm_opponent_paths, evaluate_robot_arm_league_model
from train.evaluate_robot_arm_tactical import default_tactical_opponent_paths, evaluate_robot_arm_tactical_model
from train.evaluate_selfplay import evaluate_selfplay_model
from train.evaluate_variety import evaluate_variety_model


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
    check_env(AdvancedStrokesEnv())
    check_env(CompactTechniqueEnv())
    check_env(VarietyTechniqueEnv())
    check_env(
        SelfPlayVarietyEnv(
            config=SelfPlayVarietyConfig(opponent_model_path=str(ROOT / "models" / "passed" / "ppo_stage11"))
        )
    )
    check_env(
        LeagueSelfPlayEnv(
            config=LeagueSelfPlayConfig(
                opponent_model_paths=(
                    str(ROOT / "models" / "passed" / "ppo_stage11"),
                    str(ROOT / "models" / "passed" / "ppo_stage12"),
                )
            )
        )
    )
    check_env(RobotArmPingPongEnv())
    check_env(RobotArmLeagueEnv(config=RobotArmLeagueConfig()))
    check_env(RobotArmTacticalLeagueEnv(config=RobotArmTacticalLeagueConfig()))
    check_env(RobotArmAttackLeagueEnv(config=RobotArmAttackLeagueConfig()))


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
            metrics["normal_end_rate"] >= 0.40
            and metrics["win_rate"] >= 0.40
            and metrics["avg_rally_length"] >= 10.0
            and metrics["avg_legal_landings"] >= 20.0
            and metrics["attack_attempt_rate"] >= 0.05
            and metrics["attack_success_rate"] >= 0.25
            and metrics["attack_landing_rate"] >= 0.20
            and metrics["avg_abs_spin"] >= 0.25
            and metrics["avg_paddle_x_range"] >= 35.0
            and metrics["avg_reward"] > 0.0,
            "normal_end_rate >= 0.40, win_rate >= 0.40, avg_rally_length >= 10.0, avg_legal_landings >= 20, attack_attempt_rate >= 0.05, attack_success_rate >= 0.25, attack_landing_rate >= 0.20, avg_abs_spin >= 0.25, avg_paddle_x_range >= 35, avg_reward > 0",
        )
    if stage == 9:
        return (
            metrics["normal_end_rate"] >= 0.60
            and metrics["win_rate"] >= 0.50
            and metrics["avg_rally_length"] >= 15.0
            and metrics["outside_play_rate"] >= 0.75
            and metrics["outside_hit_rate"] >= 0.50
            and metrics["loop_landing_rate"] >= 0.50
            and metrics["drive_landing_rate"] >= 0.15
            and metrics["avg_max_topspin"] >= 3.0
            and metrics["avg_paddle_x_range"] >= 120.0
            and metrics["avg_reward"] > 0.0,
            "normal_end_rate >= 0.60, win_rate >= 0.50, avg_rally_length >= 15, outside_play_rate >= 0.75, outside_hit_rate >= 0.50, loop_landing_rate >= 0.50, drive_landing_rate >= 0.15, avg_max_topspin >= 3, avg_paddle_x_range >= 120, avg_reward > 0",
        )
    if stage == 10:
        return (
            metrics["normal_end_rate"] >= 0.60
            and metrics["avg_rally_length"] >= 15.0
            and metrics["loop_landing_rate"] >= 0.75
            and metrics["avg_angled_hit_rate"] >= 0.85
            and metrics["avg_compact_technique_rate"] >= 0.85
            and metrics["compact_technique_landing_rate"] >= 0.75
            and metrics["avg_block_rate"] <= 0.05
            and metrics["avg_max_topspin"] >= 3.0
            and metrics["avg_reward"] > 0.0,
            "normal_end_rate >= 0.60, avg_rally_length >= 15, loop_landing_rate >= 0.75, avg_angled_hit_rate >= 0.85, avg_compact_technique_rate >= 0.85, compact_technique_landing_rate >= 0.75, avg_block_rate <= 0.05, avg_max_topspin >= 3, avg_reward > 0",
        )
    if stage == 11:
        return (
            metrics["normal_end_rate"] >= 0.80
            and metrics["avg_rally_length"] >= 8.0
            and metrics["loop_landing_rate"] >= 0.50
            and metrics["drive_landing_rate"] >= 0.50
            and metrics["smash_landing_rate"] >= 0.35
            and metrics["high_arc_rate"] >= 0.40
            and metrics["avg_unique_styles_landed"] >= 2.0
            and metrics["avg_style_match_landings"] >= 3.0
            and metrics["avg_max_topspin"] >= 4.0
            and metrics["avg_block_rate"] <= 0.20
            and metrics["avg_reward"] > 0.0,
            "normal_end_rate >= 0.80, avg_rally_length >= 8, loop_landing_rate >= 0.50, drive_landing_rate >= 0.50, smash_landing_rate >= 0.35, high_arc_rate >= 0.40, avg_unique_styles_landed >= 2, avg_style_match_landings >= 3, avg_max_topspin >= 4, avg_block_rate <= 0.20, avg_reward > 0",
        )
    if stage == 12:
        return (
            metrics["opponent_loaded_rate"] >= 1.0
            and metrics["normal_end_rate"] >= 0.95
            and metrics["win_rate"] >= 0.70
            and metrics["hit_rate"] >= 0.90
            and metrics["avg_rally_length"] >= 1.5
            and metrics["loop_landing_rate"] >= 0.35
            and metrics["avg_reward"] > 25.0,
            "opponent_loaded_rate >= 1.0, normal_end_rate >= 0.95, win_rate >= 0.70, hit_rate >= 0.90, avg_rally_length >= 1.5, loop_landing_rate >= 0.35, avg_reward > 25",
        )
    if stage == 13:
        return (
            metrics["opponent_pool_size"] >= 4
            and metrics["opponent_loaded_rate"] >= 1.0
            and metrics["normal_end_rate"] >= 0.95
            and metrics["pool_win_rate"] >= 0.49
            and metrics["worst_opponent_win_rate"] >= 0.45
            and metrics["hit_rate"] >= 0.95
            and metrics["avg_rally_length"] >= 3.8
            and metrics["loop_landing_rate"] >= 0.55
            and metrics["drive_landing_rate"] >= 0.50
            and metrics["estimated_elo_delta_vs_pool"] >= -10.0,
            "opponent_pool_size >= 4, opponent_loaded_rate >= 1.0, normal_end_rate >= 0.95, pool_win_rate >= 0.49, worst_opponent_win_rate >= 0.45, hit_rate >= 0.95, avg_rally_length >= 3.8, loop_landing_rate >= 0.55, drive_landing_rate >= 0.50, estimated_elo_delta_vs_pool >= -10",
        )
    if stage == 14:
        return (
            metrics["robot_arm_enabled_rate"] >= 1.0
            and metrics["normal_end_rate"] >= 0.80
            and metrics["hit_rate"] >= 0.90
            and metrics["avg_rally_length"] >= 8.0
            and metrics["loop_landing_rate"] >= 0.80
            and metrics["drive_landing_rate"] >= 0.80
            and metrics["topspin_landing_rate"] >= 0.80
            and metrics["avg_max_topspin"] >= 3.0
            and metrics["avg_tracking_error"] <= 45.0
            and metrics["avg_reward"] > 0.0,
            "robot_arm_enabled_rate >= 1.0, normal_end_rate >= 0.80, hit_rate >= 0.90, avg_rally_length >= 8, loop_landing_rate >= 0.80, drive_landing_rate >= 0.80, topspin_landing_rate >= 0.80, avg_max_topspin >= 3, avg_tracking_error <= 45, avg_reward > 0",
        )
    if stage == 15:
        return (
            metrics["opponent_pool_size"] >= 4
            and metrics["opponent_loaded_rate"] >= 1.0
            and metrics["normal_end_rate"] >= 0.95
            and metrics["pool_win_rate"] >= 0.60
            and metrics["worst_opponent_win_rate"] >= 0.55
            and metrics["hit_rate"] >= 0.90
            and metrics["avg_rally_length"] >= 8.0
            and metrics["loop_landing_rate"] >= 0.85
            and metrics["drive_landing_rate"] >= 0.85
            and metrics["topspin_landing_rate"] >= 0.85
            and metrics["avg_max_topspin"] >= 3.5
            and metrics["avg_reward"] > 0.0,
            "opponent_pool_size >= 4, opponent_loaded_rate >= 1.0, normal_end_rate >= 0.95, pool_win_rate >= 0.60, worst_opponent_win_rate >= 0.55, hit_rate >= 0.90, avg_rally_length >= 8, loop_landing_rate >= 0.85, drive_landing_rate >= 0.85, topspin_landing_rate >= 0.85, avg_max_topspin >= 3.5, avg_reward > 0",
        )
    if stage == 16:
        return (
            metrics["opponent_pool_size"] >= 5
            and metrics["opponent_loaded_rate"] >= 1.0
            and metrics["normal_end_rate"] >= 0.95
            and metrics["hit_rate"] >= 0.90
            and metrics["avg_rally_length"] >= 12.0
            and metrics["loop_landing_rate"] >= 0.90
            and metrics["drive_landing_rate"] >= 0.90
            and metrics["topspin_landing_rate"] >= 0.90
            and metrics["avg_max_topspin"] >= 3.5
            and metrics["wrong_side_score_rate"] <= 0.05
            and metrics["avg_reward"] > 20.0,
            "opponent_pool_size >= 5, opponent_loaded_rate >= 1.0, normal_end_rate >= 0.95, hit_rate >= 0.90, avg_rally_length >= 12, loop_landing_rate >= 0.90, drive_landing_rate >= 0.90, topspin_landing_rate >= 0.90, avg_max_topspin >= 3.5, wrong_side_score_rate <= 0.05, avg_reward > 20",
        )
    if stage == 17:
        return (
            metrics["opponent_pool_size"] >= 6
            and metrics["opponent_loaded_rate"] >= 1.0
            and metrics["normal_end_rate"] >= 0.95
            and metrics["pool_win_rate"] >= 0.45
            and metrics["hit_rate"] >= 0.90
            and metrics["avg_rally_length"] >= 8.0
            and metrics["avg_legal_landings"] >= 10.0
            and metrics["loop_landing_rate"] >= 0.85
            and metrics["drive_landing_rate"] >= 0.85
            and metrics["topspin_landing_rate"] >= 0.85
            and metrics["attack_landing_rate"] >= 0.85
            and metrics["high_pressure_rate"] >= 0.45
            and metrics["avg_max_attack_pressure"] >= 0.55
            and metrics["wrong_side_score_rate"] <= 0.08
            and metrics["clean_attack_score_rate"] >= 0.35
            and metrics["avg_reward"] > 22.0,
            "opponent_pool_size >= 6, opponent_loaded_rate >= 1.0, normal_end_rate >= 0.95, pool_win_rate >= 0.45, hit_rate >= 0.90, avg_rally_length >= 8, avg_legal_landings >= 10, loop_landing_rate >= 0.85, drive_landing_rate >= 0.85, topspin_landing_rate >= 0.85, attack_landing_rate >= 0.85, high_pressure_rate >= 0.45, avg_max_attack_pressure >= 0.55, wrong_side_score_rate <= 0.08, clean_attack_score_rate >= 0.35, avg_reward > 22",
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
        9: "python -m train.train_advanced --timesteps 600000",
        10: "python -m train.train_compact --timesteps 400000",
        11: "python -m train.train_variety --timesteps 300000",
        12: "python -m train.train_selfplay_stage12 --generation 1 --base-model-path models/passed/ppo_stage11 --opponent-model-path models/passed/ppo_stage11 --timesteps 200000",
        13: "python -m train.train_league_stage13 --generation 3 --base-model-path models/passed/ppo_stage13 --timesteps 200000",
        14: "python -m train.train_robot_arm --load-model-path models/ppo_robot_arm_stage14 --timesteps 300000",
        15: "python -m train.train_robot_arm_league --generation 3 --base-model-path models/passed/ppo_stage15 --timesteps 200000",
        16: "python -m train.train_robot_arm_tactical --generation 2 --base-model-path models/passed/ppo_stage16 --timesteps 120000",
        17: "python -m train.train_robot_arm_attack --generation 2 --base-model-path models/passed/ppo_stage17 --timesteps 160000",
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
        elif result.stage == 8:
            key_metrics = (
                f"win={metrics['win_rate']:.3f}, rally={metrics['avg_rally_length']:.2f}, "
                f"attempt={metrics['attack_attempt_rate']:.3f}, attack={metrics['attack_success_rate']:.3f}, "
                f"land={metrics['attack_landing_rate']:.3f}, spin={metrics['avg_abs_spin']:.2f}, "
                f"x_range={metrics['avg_paddle_x_range']:.1f}, reward={metrics['avg_reward']:.3f}"
            )
        elif result.stage == 9:
            key_metrics = (
                f"win={metrics['win_rate']:.3f}, rally={metrics['avg_rally_length']:.2f}, "
                f"outside={metrics['outside_hit_rate']:.3f}, loop={metrics['loop_landing_rate']:.3f}, "
                f"drive={metrics['drive_landing_rate']:.3f}, top={metrics['avg_max_topspin']:.2f}, "
                f"x_range={metrics['avg_paddle_x_range']:.1f}, reward={metrics['avg_reward']:.3f}"
            )
        elif result.stage == 10:
            key_metrics = (
                f"win={metrics['win_rate']:.3f}, rally={metrics['avg_rally_length']:.2f}, "
                f"angled={metrics['avg_angled_hit_rate']:.3f}, tech={metrics['avg_compact_technique_rate']:.3f}, "
                f"loop={metrics['loop_landing_rate']:.3f}, drive={metrics['drive_landing_rate']:.3f}, "
                f"block={metrics['avg_block_rate']:.3f}, top={metrics['avg_max_topspin']:.2f}, "
                f"reward={metrics['avg_reward']:.3f}"
            )
        elif result.stage == 11:
            key_metrics = (
                f"rally={metrics['avg_rally_length']:.2f}, loop={metrics['loop_landing_rate']:.3f}, "
                f"drive={metrics['drive_landing_rate']:.3f}, smash={metrics['smash_landing_rate']:.3f}, "
                f"arc={metrics['high_arc_rate']:.3f}, styles={metrics['avg_unique_styles_landed']:.2f}, "
                f"top={metrics['avg_max_topspin']:.2f}, reward={metrics['avg_reward']:.3f}"
            )
        elif result.stage == 12:
            key_metrics = (
                f"loaded={metrics['opponent_loaded_rate']:.3f}, win={metrics['win_rate']:.3f}, "
                f"hit={metrics['hit_rate']:.3f}, rally={metrics['avg_rally_length']:.2f}, "
                f"loop={metrics['loop_landing_rate']:.3f}, reward={metrics['avg_reward']:.3f}"
            )
        elif result.stage == 16:
            key_metrics = (
                f"pool={metrics['opponent_pool_size']}, rally={metrics['avg_rally_length']:.2f}, "
                f"wrong={metrics['wrong_side_score_rate']:.3f}, loop={metrics['loop_landing_rate']:.3f}, "
                f"drive={metrics['drive_landing_rate']:.3f}, top={metrics['avg_max_topspin']:.2f}, "
                f"reward={metrics['avg_reward']:.3f}"
            )
        elif result.stage == 17:
            key_metrics = (
                f"pool={metrics['opponent_pool_size']}, win={metrics['pool_win_rate']:.3f}, "
                f"rally={metrics['avg_rally_length']:.2f}, legal={metrics['avg_legal_landings']:.1f}, "
                f"attack={metrics['attack_landing_rate']:.3f}, "
                f"pressure={metrics['high_pressure_rate']:.3f}, clean_score={metrics['clean_attack_score_rate']:.3f}, "
                f"wrong={metrics['wrong_side_score_rate']:.3f}, reward={metrics['avg_reward']:.3f}"
            )
        else:
            key_metrics = (
                f"robot={metrics['robot_arm_enabled_rate']:.3f}, hit={metrics['hit_rate']:.3f}, "
                f"rally={metrics['avg_rally_length']:.2f}, loop={metrics['loop_landing_rate']:.3f}, "
                f"drive={metrics['drive_landing_rate']:.3f}, top={metrics['avg_max_topspin']:.2f}, "
                f"track={metrics['avg_tracking_error']:.1f}, "
                f"reward={metrics['avg_reward']:.3f}"
                if result.stage == 14
                else (
                    f"pool={metrics['opponent_pool_size']}, win={metrics['pool_win_rate']:.3f}, "
                    f"worst={metrics['worst_opponent_win_rate']:.3f}, rally={metrics['avg_rally_length']:.2f}, "
                    f"loop={metrics['loop_landing_rate']:.3f}, drive={metrics['drive_landing_rate']:.3f}, "
                    f"elo={metrics['estimated_elo']:.1f}"
                )
            )
        print(f"{result.stage} | {'PASS' if result.passed else 'FAIL'} | {key_metrics} | {result.reason}")

    failures = [result for result in results if not result.passed]
    if failures:
        print("\nfailed stages:")
        for failure in failures:
            print(f"stage {failure.stage}: {failure.suggestion}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all seventeen ping pong RL stages.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--stage6-episodes", type=int, default=100)
    parser.add_argument("--stage7-episodes", type=int, default=100)
    parser.add_argument("--stage8-episodes", type=int, default=100)
    parser.add_argument("--stage9-episodes", type=int, default=100)
    parser.add_argument("--stage10-episodes", type=int, default=100)
    parser.add_argument("--stage11-episodes", type=int, default=100)
    parser.add_argument("--stage12-episodes", type=int, default=100)
    parser.add_argument("--stage13-episodes", type=int, default=100)
    parser.add_argument("--stage14-episodes", type=int, default=100)
    parser.add_argument("--stage15-episodes", type=int, default=100)
    parser.add_argument("--stage16-episodes", type=int, default=100)
    parser.add_argument("--stage17-episodes", type=int, default=100)
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
        9: _preferred_model_path(9, ROOT / "models" / "ppo_advanced_stage9"),
        10: _preferred_model_path(10, ROOT / "models" / "ppo_compact_stage10"),
        11: _preferred_model_path(11, ROOT / "models" / "ppo_variety_stage11"),
        12: _preferred_model_path(12, ROOT / "models" / "ppo_selfplay_stage12"),
        13: _preferred_model_path(13, ROOT / "models" / "ppo_league_stage13"),
        14: _preferred_model_path(14, ROOT / "models" / "ppo_robot_arm_stage14"),
        15: _preferred_model_path(15, ROOT / "models" / "ppo_robot_arm_league_stage15"),
        16: _preferred_model_path(16, ROOT / "models" / "ppo_robot_arm_tactical_stage16"),
        17: _preferred_model_path(17, ROOT / "models" / "ppo_robot_arm_attack_stage17"),
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
        (9, evaluate_advanced_model(stage_models[9], args.stage9_episodes, seed=args.seed), stage_models[9]),
        (10, evaluate_compact_model(stage_models[10], args.stage10_episodes, seed=args.seed), stage_models[10]),
        (11, evaluate_variety_model(stage_models[11], args.stage11_episodes, seed=args.seed), stage_models[11]),
        (
            12,
            evaluate_selfplay_model(stage_models[12], stage_models[11], args.stage12_episodes, seed=args.seed),
            stage_models[12],
        ),
        (
            13,
            evaluate_league_model(stage_models[13], default_opponent_paths(), args.stage13_episodes, seed=args.seed),
            stage_models[13],
        ),
        (14, evaluate_robot_arm_model(stage_models[14], args.stage14_episodes, seed=args.seed), stage_models[14]),
        (
            15,
            evaluate_robot_arm_league_model(
                stage_models[15],
                default_robot_arm_opponent_paths(),
                args.stage15_episodes,
                seed=args.seed,
            ),
            stage_models[15],
        ),
        (
            16,
            evaluate_robot_arm_tactical_model(
                stage_models[16],
                default_tactical_opponent_paths(),
                args.stage16_episodes,
                seed=args.seed,
            ),
            stage_models[16],
        ),
        (
            17,
            evaluate_robot_arm_attack_model(
                stage_models[17],
                default_attack_opponent_paths(),
                args.stage17_episodes,
                seed=args.seed,
            ),
            stage_models[17],
        ),
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
