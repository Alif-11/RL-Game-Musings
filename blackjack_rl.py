# This file will contain functions to assist me in creating a Q-learning function for Blackjack.

# Required imports.
import gymnasium as gym
import ale_py
import numpy as np
gym.register_envs(ale_py)

class BlackjackQLearningAgent:
  """
  This class contains the functionalities needed to create a Blackjack Q-learning agent.
  """

  def __init__(self):
    self.training_env = gym.make("Blackjack-v1")
    self.testing_env = gym.make("Blackjack-v1")
    self.Q_function = np.zeros((32,11,2,2)) # Observation space has shape (32, 11, 2), action space has 2 actions.

  def train(self, num_episodes=100000, max_timesteps=22):
    """
    Will train our Q function on the Blackjack environment.

    - num_episodes: How many total episodes to train for.
    - max_timesteps: The maximum number of timesteps we will let each episode run for.
    """
    for episode_idx in range(num_episodes):
      obs, info = self.training_env.reset()
      timestep = 0
      print(f"Current observation: {obs}")
      while timestep < max_timesteps:
        timestep += 1
        break
      

if __name__ == "__main__":
  pass