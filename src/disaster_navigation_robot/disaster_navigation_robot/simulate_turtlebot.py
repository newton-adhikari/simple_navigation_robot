#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
import math


class TurtleBot3Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps=500):
        super().__init__()

        # --- ROS2 init ---
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

        # Reward/goal
        self.goal_position = None
        self.prev_distance_to_goal = None

        # Collision threshold (meters)
        self.collision_threshold_m = 0.20

        # Velocity limits (TurtleBot3 Burger)
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

    def _get_yaw_from_quaternion(self, q):
        # Convert quaternion to yaw
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

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
        if self.scan_data_raw is None or self.odom_data is None:
            self.scan_data_raw = np.full(360, 10.0, dtype=np.float32)
            self.scan_data_norm = (self.scan_data_raw / 10.0).astype(np.float32)

        # Randomize goal each episode
        self.goal_position = np.random.uniform(low=[-2.0, -2.0], high=[2.0, 2.0])

        # Initialize distance
        pos = self.odom_data.pose.pose.position
        current_pos = np.array([pos.x, pos.y], dtype=np.float32)
        self.prev_distance_to_goal = np.linalg.norm(self.goal_position - current_pos)

        obs = self.scan_data_norm
        info = {"distance_to_goal": float(self.prev_distance_to_goal), "goal": self.goal_position.tolist()}
        return obs, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Map normalized actions to robot commands
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

        reward = -0.01  # time penalty
        terminated = False
        truncated = False

        if self.odom_data is not None:
            pos = self.odom_data.pose.pose.position
            current_pos = np.array([pos.x, pos.y], dtype=np.float32)
            dist_to_goal = np.linalg.norm(self.goal_position - current_pos)

            # Progress reward
            if self.prev_distance_to_goal is not None:
                delta = self.prev_distance_to_goal - dist_to_goal
                reward += delta * 20.0  # stronger scaling

            # Heading alignment reward
            yaw = self._get_yaw_from_quaternion(self.odom_data.pose.pose.orientation)
            heading = math.atan2(self.goal_position[1] - current_pos[1],
                                 self.goal_position[0] - current_pos[0])
            angle_diff = abs((heading - yaw + np.pi) % (2*np.pi) - np.pi)
            reward += (np.pi - angle_diff) * 0.05

            self.prev_distance_to_goal = dist_to_goal

            if dist_to_goal < 0.30:
                reward += 100.0
                terminated = True

        # Smooth collision penalty
        min_dist_m = float(np.min(obs_raw))
        reward -= 1.0 / (min_dist_m + 1e-6)
        if min_dist_m < self.collision_threshold_m:
            reward -= 20.0
            terminated = True

        if self.step_count >= self.max_episode_steps:
            truncated = True

        info = {"distance_to_goal": self.prev_distance_to_goal,
                "goal": self.goal_position.tolist(),
                "min_lidar_distance_m": min_dist_m}
        return obs_norm, reward, terminated, truncated, info

    def close(self):
        self._publish_stop()
        self.node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    env = TurtleBot3Env()
    obs, info = env.reset()
    print("Initial obs shape:", obs.shape, "info:", info)

    terminated, truncated = False, False
    total_reward = 0.0
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Reward: {reward:.3f}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
    print("Episode finished, total reward:", total_reward)
    env.close()
