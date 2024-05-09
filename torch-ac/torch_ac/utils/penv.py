import multiprocessing
import gymnasium as gym
import random


multiprocessing.set_start_method("fork")

def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            if terminated or truncated:
                obs, _ = env.reset()
            conn.send((obs, reward, terminated, truncated, info))
        elif cmd == "reset":
            obs, _ = env.reset()
            conn.send(obs)
        else:
            raise NotImplementedError

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = multiprocessing.Pipe()
            self.locals.append(local)
            p = multiprocessing.Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()[0]] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, terminated, truncated, info = self.envs[0].step(actions[0])
        if terminated or truncated:
            obs, _ = self.envs[0].reset()
        results = zip(*[(obs, reward, terminated, truncated, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError


class ParallelEnvViz(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = multiprocessing.Pipe()
            self.locals.append(local)
            p = multiprocessing.Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        
        results = [self.envs[0].reset()[0]] + [local.recv() for local in self.locals]
        screens = list(map(lambda d: d["screen"], filter(lambda d: "screen" in d, results)))
        automaps = list(map(lambda d: d["automap"], filter(lambda d: "automap" in d, results)))
        results = {"screens": screens, "automaps": automaps}

        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action.item()))
      
        obs, reward, terminated, truncated, info = self.envs[0].step(actions[0].item())
        if terminated or truncated:
            obs, _ = self.envs[0].reset()
        
        results = zip(*[(obs, reward, terminated, truncated, info)] + [local.recv() for local in self.locals])
        results = list(results)
        screens = list(map(lambda d: d["screen"], filter(lambda d: "screen" in d, results[0])))
        automaps = list(map(lambda d: d["automap"], filter(lambda d: "automap" in d, results[0])))
        results[0] = {"screens": screens, "automaps": automaps}

        return results

    def render(self):
        raise NotImplementedError
