from gym.envs.registration import register

register(
    id="gym_examples/Model-v0",
    entry_point="gym_examples.envs:ModelEnv",
    max_episode_steps=1000,
)
