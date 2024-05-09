import gymnasium as gym
from vizdoom import gymnasium_wrapper

def make_environment(env_id):
    env = gym.make(env_id, frame_skip=4)
    return env
