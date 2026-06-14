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

