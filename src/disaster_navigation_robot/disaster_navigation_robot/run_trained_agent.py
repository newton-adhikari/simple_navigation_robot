#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3 import PPO


class TurtleBot3Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps=500):
        super().__init__()
        rclpy.init()
        self.node = rclpy.create_node('turtlebot3_env')

        # Publisher for actions
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribers
        self.scan_data_raw = None
        self.scan_data_norm = None
        self.node.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.odom_data = None
        self.node.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Gazebo reset service client
        self.reset_client = self.node.create_client(Empty, '/reset_simulation')

        # Gym spaces
        self.observation_space = Box(low=0.0, high=1.0, shape=(360,), dtype=np.float32)
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Episode control
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        # Goal and reward bookkeeping
        self.goal_position = np.array([2.0, 2.0], dtype=np.float32)
        self.prev_distance_to_goal = None

        # Collision threshold (meters)
        self.collision_threshold_m = 0.20

        # TurtleBot3 velocity limits
        self.max_lin = 0.22
        self.max_ang = 2.84

    # --- ROS Callbacks ---
    def scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[np.isinf(ranges)] = msg.range_max
        ranges = np.nan_to_num(ranges, nan=msg.range_max)

        self.scan_data_raw = ranges
        clipped = np.clip(ranges, 0.0, 10.0)
        self.scan_data_norm = (clipped / 10.0).astype(np.float32)

    def odom_callback(self, msg: Odometry):
        self.odom_data = msg

    # --- Helpers ---
    def _publish_stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

    # --- Gymnasium API ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.prev_distance_to_goal = None

        # Stop and reset simulation
        self._publish_stop()
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Waiting for /reset_simulation service...')
        req = Empty.Request()
        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

        # Wait for fresh scan and odom
        for _ in range(50):
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if self.scan_data_raw is not None and self.odom_data is not None:
                break
        if self.scan_data_raw is None:
            self.scan_data_raw = np.full(360, 10.0, dtype=np.float32)
            self.scan_data_norm = (self.scan_data_raw / 10.0).astype(np.float32)

        # Initialize goal distance
        if self.odom_data is not None:
            pos = self.odom_data.pose.pose.position
            current_pos = np.array([pos.x, pos.y], dtype=np.float32)
            self.prev_distance_to_goal = float(np.linalg.norm(self.goal_position - current_pos))

        obs = self.scan_data_norm
        info = {"distance_to_goal": self.prev_distance_to_goal}
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Map normalized actions
        lin_vel = float((action[0] + 1.0) * 0.5 * self.max_lin)
        ang_vel = float(action[1] * self.max_ang)

        twist = Twist()
        twist.linear.x = lin_vel
        twist.angular.z = ang_vel
        self.cmd_vel_pub.publish(twist)

        # Process ROS messages
        rclpy.spin_once(self.node, timeout_sec=0.1)
        self.step_count += 1

        obs_norm = self.scan_data_norm if self.scan_data_norm is not None else np.zeros(360, dtype=np.float32)
        obs_raw = self.scan_data_raw if self.scan_data_raw is not None else np.full(360, 10.0, dtype=np.float32)

        reward = -0.01
        terminated = False
        truncated = False

        # Goal progress reward
        ##TODO this needs to be improved for better reward
        if self.odom_data is not None:
            pos = self.odom_data.pose.pose.position
            current_pos = np.array([pos.x, pos.y], dtype=np.float32)
            dist_to_goal = float(np.linalg.norm(self.goal_position - current_pos))

            if self.prev_distance_to_goal is not None:
                delta = self.prev_distance_to_goal - dist_to_goal
                reward += delta * 5.0

            self.prev_distance_to_goal = dist_to_goal

            if dist_to_goal < 0.30:
                reward += 50.0
                terminated = True

        reward += 0.05  # survival bonus

        # Collision check
        min_dist_m = float(np.min(obs_raw))
        if min_dist_m < self.collision_threshold_m:
            reward -= 10.0
            terminated = True

        if self.step_count >= self.max_episode_steps:
            truncated = True

        info = {
            "distance_to_goal": self.prev_distance_to_goal,
            "min_lidar_distance_m": min_dist_m
        }
        return obs_norm, reward, terminated, truncated, info

    def close(self):
        self._publish_stop()
        self.node.destroy_node()
        rclpy.shutdown()


def main():
    env = TurtleBot3Env(max_episode_steps=20000)

    # Load trained PPO model
    model = PPO.load("ppo_turtlebot3.zip", env=env)

    obs, info = env.reset()
    terminated, truncated = False, False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print("Test episode reward:", total_reward)
    env.close()


if __name__ == "__main__":
    main()
