#!/usr/bin/env python3

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from simulate_turtlebot import TurtleBot3Env 

def main():
    env = TurtleBot3Env(max_episode_steps=500)

    # Optional: check environment compliance
    check_env(env)

    model = PPO("MlpPolicy", env, verbose=1)

    print("Starting training...")
    model.learn(total_timesteps=10000)

    # Save model
    model.save("ppo_turtlebot3")

    # Test trained model
    obs, info = env.reset()
    terminated, truncated = False, False
    total_reward = 0.0

    while not (terminated or truncated):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print("Test episode reward:", total_reward)

    env.close()

if __name__ == "__main__":
    main()
