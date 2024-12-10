import os

import gym
import gym_examples
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common import results_plotter

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# from stable_baselines3.common.policies import FeedForwardPolicy, register_policy
import torch as th

from time import process_time
from time import perf_counter

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True



def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()

def make_env(env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id) #, render_mode="human"
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init



if __name__ == "__main__":
    # Create log dir
    log_dir = "/tmp/gym/"
    # C:\tmp\gym
    os.makedirs(log_dir, exist_ok=True)

    # # Create and wrap the environment
    # env = gym.make("gym_examples/Model-v0")
    env_id = "gym_examples/Model-v0" #"InvertedPendulum-v4"
    num_cpu = 12  # Number of processes to use
    # Create the vectorized environment
    seed = np.random.randint(0,1000)
    vec_env = SubprocVecEnv([make_env(env_id, i,seed=seed) for i in range(num_cpu)])
    # Logs will be saved in log_dir/monitor.csv
    env = VecMonitor(vec_env, log_dir)

    # # Create action noise because TD3 and DDPG use a deterministic policy
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Register the policy, it will check that the name is not already taken
    # register_policy('CustomPolicy', CustomPolicy)
    policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 128]) #net_arch=dict(pi=[128], vf=[128])

    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=10000, log_dir=log_dir)
    # Create RL model
    # model = TD3('MlpPolicy', env, action_noise=action_noise, verbose=0)
    model = PPO("MlpPolicy", env, gamma=0.99, n_steps=int(12000/num_cpu), ent_coef=0.001, learning_rate=0.0003, gae_lambda=0.99, batch_size=64, n_epochs=3, stats_window_size=12, tensorboard_log ="C:/Users/soenk/gym-examples/gym_examples/envs/MagicLogs", policy_kwargs=policy_kwargs, verbose=1) #vf_coef?
    # n_steps=12000,
    # Train the agent
    p1 = perf_counter()
    t1 = process_time()
    model.learn(total_timesteps=int(500000), callback=callback) #500000
    p2 = perf_counter()
    t2 = process_time()
    print("Total training time: ", p2-p1)


    # Helper from the library
    results_plotter.plot_results(
        [log_dir], 5e6, results_plotter.X_TIMESTEPS, "3D Ball MuJoCo"
    )


    plot_results(log_dir)
