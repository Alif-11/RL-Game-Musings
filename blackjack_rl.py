# This file will contain functions to assist me in creating a Q-learning function for Blackjack.

# Required imports.
import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
gym.register_envs(ale_py)

class BlackjackQLearningAgent:
  """
  This class contains the functionalities needed to create a Blackjack Q-learning agent.
  """

  def __init__(self):
    self.training_env = gym.make("Blackjack-v1")
    self.testing_env = gym.make("Blackjack-v1")
    self.Q_function = np.zeros((32,11,2,2)) # Observation space has shape (32, 11, 2), action space has 2 actions.
  
  def visualize_Q_function(self):
    """
    This function will help visualize the Q function using matplotlib.
    """
    matrices_to_visualize = [
      self.Q_function[:,:,0,0],
      self.Q_function[:,:,0,1],
      self.Q_function[:,:,1,0],
      self.Q_function[:,:,1,1]
      ]
    
    min_value = min([mat.min() for mat in matrices_to_visualize])
    max_value = max([mat.max() for mat in matrices_to_visualize])
    fig, axs = plt.subplots(1, 4, figsize=(15, 6))

    for i in range(4):
      current_matrix = matrices_to_visualize[i]
      ax = axs[i]
      im = axs[i].imshow(current_matrix, cmap='Blues', vmin=min_value, vmax=max_value)

       # Annotate each cell with its numeric value
      for i in range(current_matrix.shape[0]):       # 32
          for j in range(current_matrix.shape[1]):   # 11
              val = current_matrix[i, j].item()
              ax.text(j, i, f"{val:.2f}", ha='center', va='center', color='black' if val < max_value/2 else 'white', fontsize=6)

    fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6)
    plt.tight_layout()
    plt.show()

  def train(self, num_episodes=300000, max_timesteps=22, early_break=False):
    """
    Will train our Q function on the Blackjack environment.

    Args:
      num_episodes: How many total episodes to train for.
      max_timesteps: The maximum number of timesteps we will let each episode run for.
      early_break: If True, stop execution of the program early. If False, do nothing.
    """

    # tqdm might be nice here
    for episode_idx in range(num_episodes):
      obs_t, info_t = self.training_env.reset()
      timestep = 0
      #print(f"Current observation: {obs_t}")
      list_of_states = [obs_t] # used to update the Q function
      while timestep < max_timesteps:
        valid_action_t = self.training_env.action_space.sample()
        obs_next_t, reward_t, terminated_t, truncated_t, info_t = self.training_env.step(valid_action_t)
        obs_t_1, obs_t_2, obs_t_3 = obs_t
        self.Q_function[obs_t_1,obs_t_2, obs_t_3, valid_action_t] = reward_t
        #print(f"One step reward: {reward_t}")
        # after you are done with this iteration, this runs
        obs_t = obs_next_t
        timestep += 1

        # Still need to implement the rest of the Q-training
        break
      if early_break:
        if (episode_idx+1) % 10 == 0:
          
          print("Episode induced break")
          break
    self.visualize_Q_function()  

if __name__ == "__main__":
  blackjack_q_learning_agent = BlackjackQLearningAgent()
  blackjack_q_learning_agent.train(early_break=False)