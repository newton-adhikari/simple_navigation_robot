# 🤖 TurtleBot3 Reinforcement Learning Navigation

This project implements a reinforcement learning (RL) navigation agent for **TurtleBot3 Burger** in **Gazebo** using **ROS2** and **Gymnasium**. The agent is trained with **Stable-Baselines3 PPO** to navigate toward a goal while avoiding obstacles.

---

## 🚀 How It Works

### Environment (`simulate_turtlebot.py`)

- **Actions:**  
  Normalized continuous actions in \([-1,1]\) mapped to robot velocities:

  - Linear velocity:

  \[
  v = \frac{a_{\text{lin}} + 1}{2} \cdot v_{\max}, \quad v_{\max} = 0.22\, \text{m/s}
  \]

  - Angular velocity:

  \[
  \omega = a_{\text{ang}} \cdot \omega_{\max}, \quad \omega_{\max} = 2.84\, \text{rad/s}
  \]

- **Observations:**  
  360-dimensional LIDAR scan, normalized:

  \[
  \text{obs}_i = \frac{\text{clip}(r_i, 0, 10)}{10}
  \]

- **Reset:**  
  - Stops robot  
  - Calls `/reset_simulation`  
  - Waits for fresh `/scan` and `/odom`  
  - Initializes distance to goal:

  \[
  d_0 = \| g - p_0 \|_2
  \]

- **Step:**  
  - Publishes velocity command  
  - Spins ROS2 once  
  - Computes reward and termination  

---

## 🧮 Mathematical Summary

The reward is shaped as:

\[
R = R_{\text{progress}} + R_{\text{heading}} + R_{\text{goal}} + R_{\text{collision}} + R_{\text{prox}} + R_{\text{time}} + R_{\text{survival}}
\]

- **Progress toward goal:**

\[
R_{\text{progress}} = \alpha \cdot (d_{t-1} - d_t), \quad \alpha \approx 5 - 20
\]

- **Heading alignment:**

\[
R_{\text{heading}} = \beta \cdot (\pi - |\Delta \theta|), \quad \Delta \theta = \text{wrap}(\phi - \theta)
\]

- **Goal bonus:**

\[
R_{\text{goal}} = B \quad \text{if } d_t < \epsilon_{\text{goal}}, \quad B \in [50, 100]
\]

- **Collision penalty:**

\[
R_{\text{collision}} = -C \quad \text{if } d_{\min} < \epsilon_{\text{collision}}, \quad C \in [10, 20]
\]

- **Smooth proximity penalty:**

\[
R_{\text{prox}} = -\frac{\gamma}{d_{\min} + \varepsilon}
\]

- **Time shaping:**

\[
R_{\text{time}} = -0.01, \quad R_{\text{survival}} = +0.05
\]

---

## 🏋️ Training (`test_agent.py`)

- Uses **PPO** with MLP policy.

- Minimal example to train:

```bash
python3 test_agent.py
```
Saves model to ppo_turtlebot3.zip.

---

## ⚠️ For meaningful results, increase timesteps:
model.learn(total_timesteps=200000)

---

## ▶️ Running Trained Agent (run_trained_agent.py)
```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

python3 run_trained_agent.py

```

## 📊 Training Recommendations
Timesteps: 200k–1M for stable navigation.

VecNormalize: Normalize observations and rewards for stability.

VecEnvs: Use DummyVecEnv or SubprocVecEnv for parallel rollouts.

Curriculum: Start with simple worlds, then increase complexity.

Randomized goals: Prevent overfitting to a single target.

---

## 📂 Project Structure
/src
├── run_trained_agent.py
├── simulate_turtlebot.py
└── test_agent.py