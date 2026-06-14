# Python + Pygame 2D 乒乓球强化学习项目规划

## 1. 项目目标

本项目目标是使用 **Python + Pygame + Gymnasium + Stable-Baselines3** 构建一个 2D 乒乓球强化学习环境，并训练一个 agent 学会接球、回球、得分，最终可扩展到自博弈和更真实的乒乓球物理。

推荐采用渐进式开发路线：

```text
先做 Pong-like 简化环境
用 Gymnasium 封装强化学习接口
Pygame 只用于可视化和人类试玩
训练时关闭渲染
先用 PPO + 离散动作
先打脚本对手
再加课程学习
最后做自博弈和真实物理
```

---

## 2. 技术栈

### 推荐组合

```text
Python
Pygame：可视化和手动试玩
Gymnasium：封装强化学习环境
Stable-Baselines3：PPO 训练
NumPy：状态计算
PyTorch：由 Stable-Baselines3 内部使用
TensorBoard：训练日志可视化
```

### 物理引擎选择

| 路线 | 推荐程度 | 说明 |
|---|---:|---|
| 自己写简单 2D 物理 | 强烈推荐先用 | 可控、简单、训练快 |
| PyMunk / Box2D | 后期再用 | 更真实，但调试和训练复杂度高 |

建议：

```text
第一阶段：自己写球和球拍的 2D 碰撞
第二阶段：再考虑 PyMunk / Box2D
```

原因是强化学习项目的第一目标是跑通训练闭环，而不是一开始追求真实物理。

---

## 3. 项目结构

建议目录结构如下：

```text
pingpong_rl/
│
├── envs/
│   ├── pingpong_env.py        # Gymnasium 环境
│   ├── physics.py             # 球、球拍、碰撞逻辑
│   ├── rules.py               # 得分、出界、回合规则
│   └── opponents.py           # 脚本 AI / 旧模型对手
│
├── render/
│   └── pygame_renderer.py     # Pygame 可视化
│
├── train/
│   ├── train_ppo.py           # PPO 训练入口
│   ├── evaluate.py            # 评估模型
│   └── self_play.py           # 后期自博弈
│
├── play/
│   └── human_play.py          # 人类试玩环境
│
├── configs/
│   └── default.yaml           # 参数配置
│
├── models/
│   └── .gitkeep               # 保存模型
│
├── logs/
│   └── .gitkeep               # TensorBoard 日志
│
└── main.py
```

---

## 4. MVP 版本设计

不要一开始做完整真实乒乓球。第一版建议做一个 Pong-like 环境。

### MVP 规则

```text
2D 平面
左边是 agent
右边是 opponent
球在矩形场地内运动
球拍只能上下移动
球碰到上下墙反弹
球碰到球拍反弹
球越过左边界，agent 失分
球越过右边界，agent 得分
```

这个版本虽然更像 Pong，但非常适合作为强化学习起点。

---

## 5. 环境参数

建议第一版使用以下参数：

```python
WIDTH = 800
HEIGHT = 400

BALL_RADIUS = 8
BALL_SPEED_X = 5
BALL_SPEED_Y = 3

PADDLE_WIDTH = 12
PADDLE_HEIGHT = 80
PADDLE_SPEED = 6

AGENT_X = 40
OPPONENT_X = WIDTH - 40
```

注意：训练时不要直接把像素值喂给 agent，而是把坐标和速度归一化到 `[-1, 1]` 左右。

---

## 6. Observation 设计

第一版 observation 用 8 维即可：

```text
ball_x
ball_y
ball_vx
ball_vy
agent_y
agent_vy
opponent_y
opponent_vy
```

推荐归一化：

```python
obs = np.array([
    ball_x / WIDTH * 2 - 1,
    ball_y / HEIGHT * 2 - 1,
    ball_vx / MAX_BALL_SPEED,
    ball_vy / MAX_BALL_SPEED,
    agent_y / HEIGHT * 2 - 1,
    agent_vy / PADDLE_SPEED,
    opponent_y / HEIGHT * 2 - 1,
    opponent_vy / PADDLE_SPEED,
], dtype=np.float32)
```

后期可以加入：

```text
ball_distance_to_agent
predicted_ball_y_at_agent_x
relative_y = ball_y - agent_y
whether_ball_moving_towards_agent
```

---

## 7. Action 设计

第一版推荐使用离散动作：

```text
0: 不动
1: 向上移动
2: 向下移动
```

Gymnasium 定义：

```python
self.action_space = gym.spaces.Discrete(3)
```

动作执行：

```python
if action == 1:
    agent_vy = -PADDLE_SPEED
elif action == 2:
    agent_vy = PADDLE_SPEED
else:
    agent_vy = 0
```

后期可以升级为连续动作：

```python
self.action_space = gym.spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(1,),
    dtype=np.float32
)
```

---

## 8. 奖励函数设计

第一版不要只使用赢输奖励，否则 agent 学习很慢。

推荐奖励：

```text
agent 接到球：+0.2
agent 得分：+1.0
agent 失分：-1.0
靠近球：小奖励
无意义移动：小惩罚
```

示例：

```python
reward = 0.0

prev_dist = abs(prev_ball_y - prev_agent_y)
curr_dist = abs(ball_y - agent_y)
reward += 0.01 * (prev_dist - curr_dist) / HEIGHT

if action != 0:
    reward -= 0.001

if agent_hit_ball:
    reward += 0.2

if agent_scores:
    reward += 1.0
    terminated = True

if agent_loses:
    reward -= 1.0
    terminated = True
```

注意：靠近球的 shaping reward 必须很小，否则 agent 可能只学会追球，而不是学会赢球。

---

## 9. Episode 设计

第一版建议每个 episode 是一分球：

```text
reset
随机发球
直到一方得分
terminated = True
```

设置最大步数：

```python
MAX_STEPS = 1000
```

如果超过最大步数还没分出胜负：

```python
truncated = True
```

---

## 10. 对手设计

第一阶段不要同时训练两个 agent，先使用脚本对手。

### 简单脚本对手

```python
def scripted_opponent(ball_y, opponent_y):
    if ball_y < opponent_y:
        return -1
    elif ball_y > opponent_y:
        return 1
    else:
        return 0
```

对手速度建议低于 agent：

```python
opponent_speed = 4
```

### 对手难度分级

| 难度 | 行为 |
|---|---|
| 0 | 随机移动 |
| 1 | 慢速追球 |
| 2 | 正常追球 |
| 3 | 带预测落点 |
| 4 | 旧版本模型 |
| 5 | 自博弈模型池 |

训练初期推荐：

```text
70% 慢速追球
30% 随机移动
```

---

## 11. 核心环境代码骨架

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PingPongEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        super().__init__()

        self.width = 800
        self.height = 400

        self.ball_radius = 8
        self.paddle_width = 12
        self.paddle_height = 80

        self.agent_x = 40
        self.opponent_x = self.width - 40

        self.paddle_speed = 6
        self.max_ball_speed = 10
        self.max_steps = 1000

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.renderer = None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.ball_x = self.width / 2
        self.ball_y = self.height / 2

        direction = self.np_random.choice([-1, 1])
        self.ball_vx = direction * self.np_random.uniform(4.0, 6.0)
        self.ball_vy = self.np_random.uniform(-3.0, 3.0)

        self.agent_y = self.height / 2
        self.opponent_y = self.height / 2
        self.agent_vy = 0.0
        self.opponent_vy = 0.0

        self.steps = 0

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        self.steps += 1

        prev_agent_y = self.agent_y
        prev_ball_y = self.ball_y

        self._apply_agent_action(action)
        self._apply_opponent_policy()
        self._update_physics()

        reward = self._compute_shaping_reward(
            prev_agent_y=prev_agent_y,
            prev_ball_y=prev_ball_y,
            action=action
        )

        terminated = False
        truncated = False

        if self.ball_x < 0:
            reward -= 1.0
            terminated = True

        elif self.ball_x > self.width:
            reward += 1.0
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _apply_agent_action(self, action):
        if action == 1:
            self.agent_vy = -self.paddle_speed
        elif action == 2:
            self.agent_vy = self.paddle_speed
        else:
            self.agent_vy = 0.0

        self.agent_y += self.agent_vy
        self.agent_y = np.clip(
            self.agent_y,
            self.paddle_height / 2,
            self.height - self.paddle_height / 2
        )

    def _apply_opponent_policy(self):
        if self.ball_y < self.opponent_y:
            self.opponent_vy = -4
        elif self.ball_y > self.opponent_y:
            self.opponent_vy = 4
        else:
            self.opponent_vy = 0

        self.opponent_y += self.opponent_vy
        self.opponent_y = np.clip(
            self.opponent_y,
            self.paddle_height / 2,
            self.height - self.paddle_height / 2
        )

    def _update_physics(self):
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        if self.ball_y <= self.ball_radius:
            self.ball_y = self.ball_radius
            self.ball_vy *= -1

        elif self.ball_y >= self.height - self.ball_radius:
            self.ball_y = self.height - self.ball_radius
            self.ball_vy *= -1

        if self._collides_with_paddle(self.agent_x, self.agent_y):
            if self.ball_vx < 0:
                self.ball_x = self.agent_x + self.paddle_width / 2 + self.ball_radius
                self.ball_vx *= -1.05

                offset = (self.ball_y - self.agent_y) / (self.paddle_height / 2)
                self.ball_vy += offset * 2.0 + self.agent_vy * 0.2

        if self._collides_with_paddle(self.opponent_x, self.opponent_y):
            if self.ball_vx > 0:
                self.ball_x = self.opponent_x - self.paddle_width / 2 - self.ball_radius
                self.ball_vx *= -1.05

                offset = (self.ball_y - self.opponent_y) / (self.paddle_height / 2)
                self.ball_vy += offset * 2.0 + self.opponent_vy * 0.2

        self.ball_vx = np.clip(self.ball_vx, -self.max_ball_speed, self.max_ball_speed)
        self.ball_vy = np.clip(self.ball_vy, -self.max_ball_speed, self.max_ball_speed)

    def _collides_with_paddle(self, paddle_x, paddle_y):
        paddle_left = paddle_x - self.paddle_width / 2
        paddle_right = paddle_x + self.paddle_width / 2
        paddle_top = paddle_y - self.paddle_height / 2
        paddle_bottom = paddle_y + self.paddle_height / 2

        ball_left = self.ball_x - self.ball_radius
        ball_right = self.ball_x + self.ball_radius
        ball_top = self.ball_y - self.ball_radius
        ball_bottom = self.ball_y + self.ball_radius

        return (
            ball_right >= paddle_left and
            ball_left <= paddle_right and
            ball_bottom >= paddle_top and
            ball_top <= paddle_bottom
        )

    def _compute_shaping_reward(self, prev_agent_y, prev_ball_y, action):
        reward = 0.0

        prev_dist = abs(prev_ball_y - prev_agent_y)
        curr_dist = abs(self.ball_y - self.agent_y)

        reward += 0.01 * (prev_dist - curr_dist) / self.height

        if action != 0:
            reward -= 0.001

        if self.ball_vx > 0:
            reward *= 0.3

        return reward

    def _get_obs(self):
        return np.array([
            self.ball_x / self.width * 2 - 1,
            self.ball_y / self.height * 2 - 1,
            self.ball_vx / self.max_ball_speed,
            self.ball_vy / self.max_ball_speed,
            self.agent_y / self.height * 2 - 1,
            self.agent_vy / self.paddle_speed,
            self.opponent_y / self.height * 2 - 1,
            self.opponent_vy / self.paddle_speed,
        ], dtype=np.float32)

    def render(self):
        pass

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
```

---

## 12. Pygame Renderer

训练时不要打开 Pygame，评估或试玩时再打开。

```python
import pygame


class PygameRenderer:
    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("2D Ping Pong RL")
        self.clock = pygame.time.Clock()

    def render(self, env, fps=60):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.fill((20, 20, 20))

        pygame.draw.line(
            self.screen,
            (80, 80, 80),
            (self.width // 2, 0),
            (self.width // 2, self.height),
            2
        )

        pygame.draw.circle(
            self.screen,
            (240, 240, 240),
            (int(env.ball_x), int(env.ball_y)),
            env.ball_radius
        )

        pygame.draw.rect(
            self.screen,
            (100, 180, 255),
            pygame.Rect(
                env.agent_x - env.paddle_width / 2,
                env.agent_y - env.paddle_height / 2,
                env.paddle_width,
                env.paddle_height
            )
        )

        pygame.draw.rect(
            self.screen,
            (255, 120, 120),
            pygame.Rect(
                env.opponent_x - env.paddle_width / 2,
                env.opponent_y - env.paddle_height / 2,
                env.paddle_width,
                env.paddle_height
            )
        )

        pygame.display.flip()
        self.clock.tick(fps)

    def close(self):
        pygame.quit()
```

在环境中接入：

```python
def render(self):
    if self.renderer is None:
        from render.pygame_renderer import PygameRenderer
        self.renderer = PygameRenderer(self.width, self.height)

    self.renderer.render(self)
```

---

## 13. PPO 训练脚本

`train/train_ppo.py`

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from envs.pingpong_env import PingPongEnv


def make_env():
    def _init():
        return PingPongEnv(render_mode=None)
    return _init


if __name__ == "__main__":
    env = PingPongEnv()
    check_env(env)

    num_envs = 8
    vec_env = SubprocVecEnv([make_env() for _ in range(num_envs)])

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="./logs/ppo_pingpong/"
    )

    model.learn(total_timesteps=2_000_000)

    model.save("models/ppo_pingpong_v1")
```

Windows 上如果 `SubprocVecEnv` 有问题，先改用：

```python
vec_env = DummyVecEnv([make_env() for _ in range(num_envs)])
```

---

## 14. 评估脚本

`train/evaluate.py`

```python
import time
from stable_baselines3 import PPO
from envs.pingpong_env import PingPongEnv


if __name__ == "__main__":
    env = PingPongEnv(render_mode="human")
    model = PPO.load("models/ppo_pingpong_v1")

    obs, info = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            time.sleep(0.5)
            obs, info = env.reset()
```

---

## 15. 人类试玩脚本

`play/human_play.py`

```python
import pygame
from envs.pingpong_env import PingPongEnv


if __name__ == "__main__":
    env = PingPongEnv(render_mode="human")
    obs, info = env.reset()

    while True:
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            action = 1
        elif keys[pygame.K_DOWN]:
            action = 2
        else:
            action = 0

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()
```

在训练前，必须先手动测试：

```text
球拍碰撞是否正常
球是否会卡进球拍
反弹角度是否合理
速度是否过快
episode 是否正确结束
```

---

## 16. 训练阶段规划

### 阶段 0：环境验证

目标：

```text
人类可以正常玩
球不会穿墙
球不会卡在球拍里
得分和 reset 正常
```

---

### 阶段 1：学会接球

先不要打完整比赛。

设计：

```text
球从右向左随机发出
agent 只要碰到球就成功
漏球就失败
```

奖励：

```text
碰球 +1
漏球 -1
靠近球 +小奖励
```

---

### 阶段 2：学会 Pong 对打

加入脚本对手。

奖励：

```text
碰球 +0.2
得分 +1
失分 -1
移动 -0.001
```

目标：

```text
agent 对慢速脚本 AI 胜率 > 70%
```

---

### 阶段 3：提高对手难度

让 opponent 预测球的落点：

```python
predicted_y = predict_ball_y_at_x(
    ball_x,
    ball_y,
    ball_vx,
    ball_vy,
    opponent_x
)
```

对手移动到预测落点，而不是追当前 `ball_y`。

---

### 阶段 4：课程学习

逐渐增加难度：

```text
初期球速慢
agent 胜率上升后，球速变快
初期对手慢
agent 胜率上升后，对手变快
初期球拍大
agent 胜率上升后，球拍变小
```

推荐参数：

| 阶段 | 球速 | 对手速度 | 球拍高度 |
|---|---:|---:|---:|
| Easy | 3-4 | 2 | 100 |
| Medium | 4-6 | 4 | 80 |
| Hard | 6-8 | 5 | 60 |
| Expert | 8-10 | 6 | 50 |

---

### 阶段 5：自博弈

当 PPO 能稳定打败脚本对手后，再做自博弈。

基本思路：

```text
当前模型训练一段时间
保存 checkpoint
之后随机用旧 checkpoint 当 opponent
```

模型池：

```text
models/opponents/
├── ppo_100k.zip
├── ppo_300k.zip
├── ppo_500k.zip
└── ppo_1m.zip
```

环境中的 opponent 类型：

```python
opponent_type = "scripted"
opponent_type = "random"
opponent_type = "model"
```

---

## 17. 推荐训练路线图

| 阶段 | 环境 | 动作 | 对手 | 训练步数 |
|---|---|---|---|---:|
| 1 | 单侧接球 | 离散 | 无 | 100k - 300k |
| 2 | 简单 Pong | 离散 | 慢速脚本 | 500k - 1M |
| 3 | Pong 加速 | 离散 | 正常脚本 | 1M - 3M |
| 4 | 预测型对手 | 离散 | 预测脚本 | 3M - 5M |
| 5 | 自博弈 | 离散/连续 | 旧模型池 | 5M+ |
| 6 | 真实乒乓球规则 | 连续 | 自博弈 | 10M+ |

---

## 18. 日志和评估指标

不要只看 reward。建议记录：

```text
episode_reward
agent_win_rate
agent_hit_rate
average_rally_length
average_ball_speed
agent_miss_count
opponent_miss_count
```

可以在 `info` 中返回：

```python
info = {
    "agent_hit": agent_hit_ball,
    "agent_score": agent_scores,
    "agent_miss": agent_loses,
    "rally_length": self.rally_length,
}
```

---

## 19. 后期升级方向

### 19.1 连续动作

从：

```python
spaces.Discrete(3)
```

升级到：

```python
spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
```

动作含义：

```text
agent_vy = action[0] * paddle_speed
```

---

### 19.2 球拍角度

动作变成：

```text
move_y
paddle_angle
```

```python
spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
```

反弹时根据球拍角度改变球方向。

---

### 19.3 球拍挥拍速度

动作变成：

```text
move_y
swing
angle
```

此时更接近真实乒乓球。

---

### 19.4 加入桌面和球网

从 Pong 升级为侧视乒乓：

```text
球有重力
球会落台弹起
球必须过网
球必须落在对方桌面
```

这一步难度会明显提高，建议等 Pong 版本训练稳定后再做。

---

## 20. 关键实现建议

### 20.1 训练时关闭 Pygame

```python
env = PingPongEnv(render_mode=None)
```

评估时再开：

```python
env = PingPongEnv(render_mode="human")
```

---

### 20.2 固定随机种子调试

```python
obs, info = env.reset(seed=42)
```

这样 bug 更容易复现。

---

### 20.3 step 中保持 deterministic

不要在 `step()` 中随便加随机噪声。

随机性应放在：

```text
reset()
domain randomization
curriculum 参数初始化
```

---

### 20.4 碰撞后把球推出球拍

否则球可能卡在 paddle 里反复碰撞。

```python
self.ball_x = self.agent_x + self.paddle_width / 2 + self.ball_radius
```

---

### 20.5 先不要用 CNN

不要一开始从 Pygame 图像训练。

推荐先使用结构化 observation。等 agent 已经能学会后，再尝试 pixel-based RL。

---

## 21. 最推荐的开发顺序

```text
1. 写 PingPongEnv，不带 Pygame
2. 写 physics：球运动、墙反弹、球拍碰撞
3. 写 PygameRenderer
4. 写 human_play.py，自己手动玩
5. 写 train_ppo.py，跑 PPO
6. 写 evaluate.py，看模型表现
7. 加日志：胜率、碰球率、回合长度
8. 调 reward
9. 加课程学习
10. 加更强对手
11. 加自博弈
12. 再考虑真实乒乓球规则
```

---

## 22. 第一个里程碑

建议第一个可运行目标：

```text
一个 agent 使用 PPO，在简单 Pong 环境中，
经过 1M steps 后，
能稳定接住脚本对手打来的球，
并且胜率超过 60%。
```

这个目标可以验证完整链路：

```text
环境 → Pygame → Gymnasium → PPO → 模型保存 → 模型评估
```

---

## 23. 最小依赖安装

```bash
pip install gymnasium stable-baselines3 pygame numpy tensorboard
```

训练：

```bash
python train/train_ppo.py
```

查看 TensorBoard：

```bash
tensorboard --logdir logs
```

评估：

```bash
python train/evaluate.py
```

---

## 24. 总结

最佳路线：

```text
先做 Pong-like 简化环境
用 Gymnasium 封装
Pygame 只用于人类试玩和评估
训练时关闭渲染
先用 PPO + 离散动作
先打脚本对手
再加课程学习
最后做自博弈和真实乒乓球物理
```

项目早期不要追求真实。强化学习项目最重要的是先把训练闭环跑通：

```text
状态 observation
动作 action
奖励 reward
reset
step
模型训练
模型评估
```

只要这个闭环稳定，后面再逐步增加复杂物理、连续动作、球拍角度、球网、桌面弹跳和自博弈。
