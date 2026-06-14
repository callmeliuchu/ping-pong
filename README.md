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

## Validate All Stages

Run the full seven-stage gate check:

```bash
python -m train.validate_stages --episodes 200 --stage6-episodes 100 --stage7-episodes 100
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
```

Validation metrics are written to `logs/validation/stage*.json`.
