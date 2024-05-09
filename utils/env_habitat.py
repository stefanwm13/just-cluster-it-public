import os
import pathlib

import gym
import habitat
from habitat_baselines.config.default import get_config
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.utils.env_utils import make_env_fn
from habitat.utils.visualizations.utils import observations_to_image
from habitat.datasets.pointnav.pointnav_generator import is_compatible_episode

import torch
import numpy as np
def _format_observation(obs):
    obs = torch.squeeze(torch.tensor(obs))
    return obs.view((1, 1) + obs.shape)

def _sample_start_and_goal(sim, seed, number_retries_per_target=100):
    sim.seed(0) # Target is always the same
    target_position = sim.sample_navigable_point()
    sim.seed(seed) # Start depends on the seed
    for _retry in range(number_retries_per_target):
        source_position = sim.sample_navigable_point()
        is_compatible, _ = is_compatible_episode(
            source_position,
            target_position,
            sim,
            near_dist=1,
            far_dist=30,
            geodesic_to_euclid_ratio=1.1,
        )
        if is_compatible:
            break
    if not is_compatible:
        raise ValueError('Cannot find a goal position.')
    return source_position, target_position


def make_gym_env(env_id, seed=0):
    config_file = os.path.join(pathlib.Path(__file__).parent.resolve(),
                               'habitat_config',
                               'pointnav_apartment-0.yaml')  # Absolute path

    print(config_file)

    config = get_config(config_paths=config_file, opts=['BASE_TASK_CONFIG_PATH', config_file])

    config.defrost()

    # Overwrite all RGBs width / height of TASK (not SIMULATOR)ff
    for k in config['TASK_CONFIG']['SIMULATOR']:
        if 'rgb' in k.lower():
            config['TASK_CONFIG']['SIMULATOR'][k]['HEIGHT'] = 64
            config['TASK_CONFIG']['SIMULATOR'][k]['WIDTH'] = 64

    # Set Replica scene
    scene = env_id[len('HabitatNav-'):]
    assert len(scene) > 0, 'Undefined scene.'
    config.TASK_CONFIG.DATASET.SCENES_DIR += scene

    config.freeze()

    # Make env
    env_class = get_env_class(config.ENV_NAME)
    env = make_env_fn(env_class=env_class, config=config)

    # Sample and set goal position
    source_location, goal_location = _sample_start_and_goal(env._env._sim, seed)
    env._env._dataset.episodes[0].start_position = source_location  # Depends on seed
    env._env._dataset.episodes[0].goals[0].position = goal_location  # Fixed

    env = HabitatNavigationWrapper(env)
    env.seed(seed)

    return env


def make_environment(env_name, actor_id=1):
    print("TEST ENV")
    env = make_gym_env(env_name, 1337)
    print(env)
    return EnvironmentHabitat(env, no_task=False, namefile="test")


class HabitatNavigationWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.action_space = gym.spaces.Discrete(env.action_space.n - 1)
        self.observation_space = self.env.observation_space['rgb']
        self._last_full_obs = None
        self._viewer = None

    def reset(self):
        obs = self.env.reset()
        self._last_full_obs = obs
        return np.asarray(obs['rgb'])

    def get_position(self):
        return self.env._env._sim.get_agent_state().position

    def step(self, action):
        obs, rwd, done, info = self.env.step(**{'action': action + 1})
        self._last_full_obs = obs
        obs = np.asarray(obs['rgb'])
        info.update({'position': self.get_position()})
        return obs, rwd, done, info


class EnvironmentHabitat:
    def __init__(self, gym_envs, no_task=False, namefile=''):
        self.all_envs = gym_envs
        # self.env_iter = itertools.cycle(gym_envs)
        self.gym_env = gym_envs
        self.episode_return = None
        self.episode_step = None
        self.no_task = True
        self.true_state_count = dict()  # Count (x,y) position (the true state)
        self.namefile = namefile
        self.reward = None
        self.frame = None
        self.done = None
        self.info = None

    def render(self, mode='rgb_array', dt=10):
        if mode == "rgb_array":
            frame = observations_to_image(
                self.gym_env._last_full_obs, self.gym_env.unwrapped._env.get_metrics()
            )
        else:
            raise ValueError(f"Render mode {mode} not currently supported.")
        if self.gym_env._viewer is None:
            self.gym_env._viewer = ImageViewer(self.gym_env.observation_space[0:2], dt)
        self.gym_env._viewer.display(frame)

    def initial(self):
        initial_reward = torch.zeros(1, 1)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        initial_done = torch.tensor(False, dtype=torch.bool).view(1, 1)
        initial_real_done = torch.tensor(False, dtype=torch.bool).view(1, 1)
        initial_frame = self.gym_env.reset()

        return dict(
            frame=initial_frame,
            reward=initial_reward,
            done=initial_done,
            real_done=initial_real_done,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            episode_win=torch.zeros(1, 1, dtype=torch.int32),
            interactions=torch.zeros(1, 1, dtype=torch.int32),
            visited_states=torch.tensor(len(self.true_state_count)).view(1, 1),
        )

    def step(self, action):
        self.frame, self.reward, self.done, self.info = self.gym_env.step(action)

        # Count true states
        position = np.round(np.round(self.info['position'], 2) * 20) / 20
        true_state_key = tuple([position[0], position[2]])
        if true_state_key in self.true_state_count:
            self.true_state_count[true_state_key] += 1
        else:
            self.true_state_count.update({true_state_key: 1})

        self.episode_step += 1
        episode_step = self.episode_step

        self.episode_return += self.reward
        episode_return = self.episode_return

        real_done = self.done  # TODO: depends on Habitat task
        if self.no_task:
            done = self.gym_env.unwrapped._env._elapsed_steps >= self.gym_env.unwrapped._env._max_episode_steps
            real_done = False

        if self.done:
            frame = self.gym_env.reset()

        self.frame = self.frame
        self.reward = torch.tensor(self.reward).view(1, 1)
        self.done = torch.tensor(self.done, dtype=torch.bool).view(1, 1)
        real_done = torch.tensor(real_done, dtype=torch.bool).view(1, 1)


        return dict(
            frame=self.frame,
            reward=self.reward,
            done=self.done,
            real_done=real_done,
            episode_return=episode_return,
            episode_step=episode_step,
            episode_win=torch.zeros(1, 1, dtype=torch.int32),
            interactions=torch.zeros(1, 1, dtype=torch.int32),
            visited_states=torch.tensor(len(self.true_state_count)).view(1, 1),
        )

    def close(self):
        for e in self.all_envs:
            e.close()
            if e._viewer is not None:
                e._viewer.close()
