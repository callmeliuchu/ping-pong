# Ping Pong RL

Stage 1 implements a Gymnasium catch environment: a ball starts on the right,
moves left, and the agent controls one paddle to catch it.

## Conda

```bash
conda activate pingpong-rl
```

## Train Stage 1

```bash
python -m train.train_stage1 --timesteps 300000
```

## Evaluate Stage 1

```bash
python -m train.evaluate_stage1 --episodes 100
```

Render evaluation:

```bash
python -m train.evaluate_stage1 --render --episodes 10
```

## Human Play

```bash
python -m play.human_stage1
```

## Stages 2-4: Pong Opponents

Stage 2 uses a slow scripted opponent, stage 3 uses a faster scripted opponent,
and stage 4 uses a predictive opponent.

```bash
python -m train.train_pong --stage 2 --timesteps 500000
python -m train.evaluate_pong --stage 2 --episodes 100

python -m train.train_pong --stage 3 --load-model-path models/ppo_pong_stage2 --timesteps 1000000
python -m train.evaluate_pong --stage 3 --episodes 100

python -m train.train_pong --stage 4 --load-model-path models/ppo_pong_stage3 --timesteps 1500000
python -m train.evaluate_pong --stage 4 --episodes 100
```

Each training command also writes a final evaluation JSON by default under
`logs/ppo_pong_stage*/final_eval.json`. You can override it:

```bash
python -m train.train_pong --stage 2 --eval-json-path logs/stage2_eval.json
```

Render a trained Pong policy:

```bash
python -m train.evaluate_pong --stage 2 --render --episodes 0
```

Manual play:

```bash
python -m play.human_pong --stage 2
```

## Stage 5: Self-Play Bootstrap

Train against a previous model used as the opponent:

```bash
python -m train.self_play_stage5 --base-model-path models/ppo_pong_stage4 --timesteps 500000
python -m train.evaluate_pong --stage 5 --model-path models/ppo_pong_stage5 --opponent-model-path models/ppo_pong_stage4 --episodes 100
```

## Stage 6: Rule-Based Gravity Rally

Stage 6 is a side-view gravity/table/net environment with simplified table
tennis rules. A legal return must land on the opponent's side before it can be
hit, volleys are illegal, net faults/out balls are scored against the last
hitter, and a sustainable rally pass requires the opening serve landing plus 20
legal rally returns.

Smoke-test the environment:

```bash
python -m train.smoke_gravity --episodes 10
python -m train.smoke_gravity --render --episodes 0
python -m play.watch_stage6
```

Train and watch a PPO policy for the rules environment:

```bash
python -m train.train_gravity --timesteps 500000
python -m train.evaluate_gravity --episodes 100
python -m train.evaluate_gravity --render --episodes 0
```

The trained rules model is available at:

```text
models/passed/ppo_stage6_rules.zip
models/passed/ppo_stage6.zip
```

## Stage 7: Realistic Paddle and Spin

Stage 7 keeps the Stage 6 rule checks and adds a more human-like control model:
the paddle can move forward/back and up/down, rotate its face, generate spin on
contact, and the ball reacts to spin after table bounces. The PPO action is a
continuous three-axis residual over a low-level tracking controller, which makes
the agent train on realistic timing and paddle-face adjustments instead of just
teleporting to the ball.

Train and watch the Stage 7 policy:

```bash
python -m train.train_realistic --timesteps 500000
python -m train.evaluate_realistic --episodes 100
python -m train.evaluate_realistic --render --episodes 0
python -m play.watch_stage7
```

The trained Stage 7 model is available at:

```text
models/passed/ppo_stage7.zip
```

## Stage 8: Competitive Realistic Point Play

Stage 8 turns the realistic physics prototype into competitive point play. The
agent keeps the Stage 7 forward/back movement, vertical movement, paddle
rotation, and spin control, then adds an attack-intent action. The opponent has
randomized difficulty levels, moves forward/back as well as vertically, and is
stable enough that the agent must build rallies and win points with deeper,
spinnier attacks instead of waiting for easy misses.

Train and watch the Stage 8 policy:

```bash
python -m train.train_competitive --timesteps 500000
python -m train.evaluate_competitive --episodes 100
python -m train.evaluate_competitive --render --episodes 0
python -m play.watch_stage8
```

The trained Stage 8 model is available at:

```text
models/passed/ppo_stage8.zip
```

## Stage 9: Advanced Strokes

Stage 9 expands the Stage 8 competitive setup into a more realistic stroke
prototype. The paddle can move outside the table edge, recover from farther
positions, and use continuous power plus brush controls to produce drives,
loops, smashes, and chops. Stage 9 also flips the advanced topspin flight model
so positive topspin pulls the ball down into a loop arc instead of floating.

Train and watch the Stage 9 policy:

```bash
python -m train.train_advanced --timesteps 600000
python -m train.evaluate_advanced --episodes 100
python -m train.evaluate_advanced --render --episodes 0
python -m play.watch_stage9
```

The trained Stage 9 model is available at:

```text
models/passed/ppo_stage9.zip
```

## Stage 10: Compact Paddle Technique

Stage 10 keeps the Stage 9 stroke controls but uses a shorter, more realistic
paddle. A loop only counts as compact technique when the left-side agent uses a
closed racket face: the paddle top tilts forward toward the opponent, forming
the acute angle used for topspin instead of the previous open/obtuse face. The
gate checks that closed-angle contacts dominate, block returns stay low, and
the compact paddle can still build topspin loop rallies.

The spin model is still a 2D side-view approximation. Contact creates
`ball_spin`; in Stage 9 and 10 positive topspin adds downward acceleration, so
the ball can arc down into the table. It does not model full 3D sidespin or
fluid dynamics.

Train and watch the Stage 10 policy:

```bash
python -m train.train_compact --timesteps 400000
python -m train.evaluate_compact --episodes 100
python -m train.evaluate_compact --render --episodes 0
python -m play.watch_stage10
```

The trained Stage 10 model is available at:

```text
models/passed/ppo_stage10.zip
```

## Stage 11: Varied Stroke Paths

Stage 11 focuses on making the rally visually less uniform. It keeps the
compact closed-racket technique, then rotates the low-level stroke target
between high loop, drive, loop, and smash. The renderer also draws a short ball
trail so the arc and speed differences are visible in Pygame.

The Stage 11 gate checks that the policy produces multiple landed stroke types,
including high arcs, drives, and smashes, instead of only repeating the safest
loop.

Train and watch the Stage 11 policy:

```bash
python -m train.train_variety --timesteps 300000
python -m train.evaluate_variety --episodes 100
python -m train.evaluate_variety --render --episodes 0
python -m play.watch_stage11
```

The trained Stage 11 model is available at:

```text
models/passed/ppo_stage11.zip
```

## Stage 12: League Self-Play Bootstrap

Stage 12 starts the growth loop from the latest Stage 11 technique model. The
agent trains against a historical PPO opponent loaded from the league archive,
then saves the new generation back into `models/selfplay/stage12/`. This is a
bootstrap self-play loop: each generation can use the previous generation as
both the starting policy and the opponent, so the agent repeatedly learns to
beat its last version.

The current Stage 12 gate checks that the historical model is really loaded,
the new policy wins reliably against Stage 11, and the rally still includes
basic topspin loop landings. It is not yet a full Elo league with many
opponents and automatic promotion/demotion.

Train, evaluate, and watch the Stage 12 policy:

```bash
python -m train.train_selfplay_stage12 --generation 1 --base-model-path models/passed/ppo_stage11 --opponent-model-path models/passed/ppo_stage11 --timesteps 200000
python -m train.evaluate_selfplay --model-path models/passed/ppo_stage12 --opponent-model-path models/passed/ppo_stage11 --episodes 100
python -m train.evaluate_selfplay --model-path models/passed/ppo_stage12 --opponent-model-path models/passed/ppo_stage11 --render --episodes 0
python -m play.watch_stage12
```

Continue with the next generation:

```bash
python -m train.train_selfplay_stage12 --generation 2 --base-model-path models/passed/ppo_stage12 --opponent-model-path models/selfplay/stage12/gen_1 --model-path models/ppo_selfplay_stage12_gen2 --timesteps 200000
```

The trained Stage 12 model is available at:

```text
models/passed/ppo_stage12.zip
models/selfplay/stage12/gen_1.zip
```

## Stage 13: Multi-Model League Self-Play

Stage 13 upgrades the bootstrap loop into a small league. Instead of training
against only the previous generation, each episode samples from a pool of
historical PPO opponents: Stage 10, Stage 11, Stage 12 generation 1, and the
passed Stage 12 model. The league evaluator reports per-opponent win rates, a
candidate-vs-pool win-rate matrix, the weakest opponent matchup, and an
estimated Elo score derived from the win rates.

Stage 13 also discourages short-point exploits. Historical opponents use a more
stable safe-return model after contact, and the reward penalizes winning before
a six-hit rally while rewarding opponent returns and sustained wins. This makes
the model train against a harder, longer-rally pool instead of repeatedly
forcing early net faults.

The physics now treats the net as a collision object instead of an instant
fault trigger. A ball that clips the top of the net can slow down, pop up, and
continue to the other side; a lower net impact rebounds from the net face. The
point is decided later by normal landing/out/second-bounce rules.

Train, evaluate, and watch the Stage 13 policy:

```bash
python -m train.train_league_stage13 --generation 2 --base-model-path models/passed/ppo_stage12 --timesteps 200000
python -m train.evaluate_league_stage13 --model-path models/passed/ppo_stage13 --episodes 100
python -m play.watch_stage13
```

Continue the league with the next generation:

```bash
python -m train.train_league_stage13 --generation 3 --base-model-path models/passed/ppo_stage13 --timesteps 200000
```

Run the promotion-based evolution loop:

```bash
python -m train.evolve_league_stage13 --start-generation 3 --generations 1 --timesteps-per-generation 200000
```

The evolution loop evaluates the current champion, trains a candidate against a
pool that includes the champion, and promotes only candidates whose pool score,
weakest-opponent win rate, and rally length beat the current champion.

The trained Stage 13 model is available at:

```text
models/passed/ppo_stage13.zip
models/selfplay/stage13/gen_2.zip
```

## Stage 14: Joint-Controlled Robot Arm

Stage 14 leaves the previous stages unchanged and introduces a new environment
where the agent no longer commands paddle position directly. The action is the
joint velocity of a fixed three-joint robot arm: shoulder, elbow, and wrist.
The paddle is attached to the arm's wrist, so contact timing, paddle position,
and paddle angle all come from forward kinematics. The right-side opponent also
uses a fixed robot arm, driven by a scripted IK controller for this first
prototype.

The current Stage 14 policy also trains a more technical contact model. A
closed, acute racket face plus upward wrist/arm brush creates topspin; positive
topspin adds downward acceleration, so the ball can pull into a visible loop
arc. Forward arm motion is counted as a drive even when it is also a topspin
loop, and backspin/chop remains available through open-face downward brushing.
The passed model is tuned for sustained topspin/drive rallies rather than only
soft blocking.

Train, evaluate, and watch the Stage 14 policy:

```bash
python -m train.train_robot_arm --load-model-path models/passed/ppo_stage14 --timesteps 400000 --model-path models/ppo_robot_arm_stage14_spin_v2
python -m train.evaluate_robot_arm --model-path models/passed/ppo_stage14 --episodes 100
python -m play.watch_stage14 --model-path models/passed/ppo_stage14 --seed 42
```

Export a Pygame-rendered GIF:

```bash
SDL_VIDEODRIVER=dummy python -m play.watch_stage14 --model-path models/passed/ppo_stage14 --seed 42 --gif-path logs/ppo_robot_arm_stage14/stage14_robot_arm_spin.gif --max-steps 900
```

The current Stage 14 model is available at:

```text
models/passed/ppo_stage14.zip
models/ppo_robot_arm_stage14_spin_v2.zip
models/best/stage14/best_model.zip
```

## Stage 15: Robot-Arm League Self-Play

Stage 15 keeps the Stage 14 joint-controlled robot arm and replaces the
single scripted red-side opponent with a league bootstrap. The red side can
load historical robot-arm PPO models from a pool, mirror the observation into
its own shoulder/elbow/wrist action space, and add model-driven residual
stroke control over an IK tracking controller. This makes the opponent a
repeatable model-pool challenger instead of only a fixed warm-up script.

The Stage 15 gate checks that the blue champion beats a multi-model pool while
preserving sustained topspin/drive rallies. The current passed model averages
8+ rally hits against the pool, loads red-side model opponents every episode,
and keeps loop, drive, and topspin landing rates above 0.90.

Train, evaluate, evolve, and watch Stage 15:

```bash
python -m train.train_robot_arm_league --generation 2 --base-model-path models/passed/ppo_stage15 --timesteps 150000
python -m train.evaluate_robot_arm_league --model-path models/passed/ppo_stage15 --episodes 100
python -m train.evolve_robot_arm_league --generation 3 --base-model-path models/passed/ppo_stage15 --timesteps 200000
python -m play.watch_stage15 --model-path models/passed/ppo_stage15 --seed 0
```

Export a Pygame-rendered league GIF:

```bash
SDL_VIDEODRIVER=dummy python -m play.watch_stage15 --model-path models/passed/ppo_stage15 --seed 0 --gif-path logs/ppo_robot_arm_league_stage15/stage15_robot_arm_league.gif --max-steps 1400
```

The current Stage 15 model is available at:

```text
models/passed/ppo_stage15.zip
models/selfplay/stage15/gen_2.zip
models/best/stage15/best_model.zip
```

## Stage 16: Clean Tactical Robot-Arm League

Stage 16 keeps the Stage 15 robot-arm league but changes the objective from
winning through red-side wrong-side landing faults to building cleaner tactical
rallies. Red returns use a spin-aware landing calculation, so topspin no longer
falls short as often, and blue receives a penalty when it wins through red
wrong-side gifts. The result is a lower immediate win rate but a much cleaner
training target: longer exchanges, stable loop/drive/topspin landings, and very
few wrong-side score endings.

Train, evaluate, and watch Stage 16:

```bash
python -m train.train_robot_arm_tactical --generation 1 --base-model-path models/passed/ppo_stage15 --timesteps 120000
python -m train.evaluate_robot_arm_tactical --model-path models/passed/ppo_stage16 --episodes 100
python -m play.watch_stage16 --model-path models/passed/ppo_stage16 --seed 0
```

Export a Pygame-rendered tactical GIF:

```bash
SDL_VIDEODRIVER=dummy python -m play.watch_stage16 --model-path models/passed/ppo_stage16 --seed 0 --gif-path logs/ppo_robot_arm_tactical_stage16/stage16_robot_arm_tactical.gif --max-steps 1600
```

The current Stage 16 model is available at:

```text
models/passed/ppo_stage16.zip
models/selfplay/stage16/gen_1.zip
models/best/stage16/best_model.zip
```

## Validate All Stages

Run the full sixteen-stage gate check:

```bash
python -m train.validate_stages --episodes 200 --stage6-episodes 100 --stage7-episodes 100 --stage8-episodes 100 --stage9-episodes 100 --stage10-episodes 100 --stage11-episodes 100 --stage12-episodes 100 --stage13-episodes 100 --stage14-episodes 100 --stage15-episodes 100 --stage16-episodes 100
```

Passing models are copied to:

```text
models/passed/ppo_stage1.zip
models/passed/ppo_stage2.zip
models/passed/ppo_stage3.zip
models/passed/ppo_stage4.zip
models/passed/ppo_stage5.zip
models/passed/ppo_stage6.zip
models/passed/ppo_stage7.zip
models/passed/ppo_stage8.zip
models/passed/ppo_stage9.zip
models/passed/ppo_stage10.zip
models/passed/ppo_stage11.zip
models/passed/ppo_stage12.zip
models/passed/ppo_stage13.zip
models/passed/ppo_stage14.zip
models/passed/ppo_stage15.zip
models/passed/ppo_stage16.zip
```

Validation metrics are written to `logs/validation/stage*.json`.
