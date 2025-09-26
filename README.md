# 🤖 TurtleBot3 Reinforcement Learning Navigation

This project implements a reinforcement learning (RL) navigation agent for **TurtleBot3 Burger** in **Gazebo** using **ROS2** and **Gymnasium**. The agent is trained with **Stable-Baselines3 PPO** to navigate toward a goal while avoiding obstacles.

---

## 🚀 How It Works

### Environment (`simulate_turtlebot.py`)

- **Actions:**  
  Continuous actions normalized to `[-1,1]`, mapped to robot velocities:

  - Linear velocity:  
    `v = (a_lin + 1)/2 * v_max,   v_max = 0.22 m/s`

  - Angular velocity:  
    `ω = a_ang * ω_max,   ω_max = 2.84 rad/s`

- **Observations:**  
  360‑dimensional LIDAR scan, normalized:  
  `obs_i = clip(r_i, 0, 10) / 10`

- **Reset:**  
  - Stops the robot  
  - Calls `/reset_simulation`  
  - Waits for fresh `/scan` and `/odom`  
  - Initializes distance to goal:  
    `d0 = || g - p0 ||_2`

- **Step:**  
  - Publishes velocity command  
  - Spins ROS2 once  
  - Computes reward and termination  

---

## 🧮 Reward Function (Mathematical Summary)

The total reward is shaped as:

`R = R_progress + R_heading + R_goal + R_collision + R_prox + R_time + R_survival`

- **Progress toward goal:**  
  `R_progress = α * (d_{t-1} - d_t),   α ≈ 5–20`

- **Heading alignment:**  
  `R_heading = β * (π - |Δθ|),   Δθ = wrap(φ - θ)`

- **Goal bonus:**  
  `R_goal = B   if d_t < ε_goal,   B ∈ [50, 100]`

- **Collision penalty:**  
  `R_collision = -C   if d_min < ε_collision,   C ∈ [10, 20]`

- **Smooth proximity penalty:**  
  `R_prox = -γ / (d_min + ε)`

- **Time shaping:**  
  `R_time = -0.01,   R_survival = +0.05`

---

## 🔣 Symbols

Here’s what each symbol means in the equations above:

- `v` → Linear velocity of the robot (m/s)  
- `ω` → Angular velocity of the robot (rad/s)  
- `a_lin` → Normalized linear action chosen by the agent (`[-1,1]`)  
- `a_ang` → Normalized angular action chosen by the agent (`[-1,1]`)  
- `v_max` → Maximum forward speed of TurtleBot3 Burger (`0.22 m/s`)  
- `ω_max` → Maximum angular speed of TurtleBot3 Burger (`2.84 rad/s`)  
- `obs_i` → Normalized LIDAR reading for beam `i`  
- `r_i` → Raw LIDAR distance reading for beam `i` (meters)  
- `d_t` → Distance from robot to goal at timestep `t`  
- `d_{t-1}` → Distance to goal at previous timestep  
- `d0` → Initial distance to goal at episode start  
- `g` → Goal position vector `[g_x, g_y]`  
- `p0` → Initial robot position vector `[x_0, y_0]`  
- `Δθ` → Difference between robot’s heading and goal direction (radians)  
- `φ` → Bearing angle from robot to goal (radians)  
- `θ` → Robot’s current yaw angle (radians)  
- `d_min` → Minimum LIDAR distance (closest obstacle in meters)  
- `ε_goal` → Goal threshold distance (≈ 0.30 m)  
- `ε_collision` → Collision threshold distance (≈ 0.20 m)  
- `α` → Scaling factor for progress reward (≈ 5–20)  
- `β` → Scaling factor for heading alignment reward  
- `B` → Goal completion bonus (≈ 50–100)  
- `C` → Collision penalty (≈ 10–20)  
- `γ` → Scaling factor for smooth proximity penalty  
- `ε` → Small constant to avoid division by zero  
- `R_time` → Small negative reward per step (time penalty)  
- `R_survival` → Small positive reward per step (survival bonus)  

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
```plaintext
/src
├── run_trained_agent.py
├── simulate_turtlebot.py
└── test_agent.py
```
