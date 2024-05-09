import pickle
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import random

import torch
from torch_ac.format import default_preprocess_obss, RNDModel
from torch_ac.utils import DictList, ParallelEnvViz
import torchvision.transforms as T

import numpy as np
import hubconf
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity
import wandb

from collections import Counter
from itertools import groupby


class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, no_logging_wandb):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters

        self.env = ParallelEnvViz(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward
        self.no_logging_wandb = no_logging_wandb

        #self.screen = pf.screen(np.zeros((64, 64, 3)), "test")

        self.transform_atari = T.Compose([
            T.Resize((42, 42), interpolation=T.InterpolationMode.BICUBIC),
        ])

        self.transform_dino = T.Compose([
            T.Resize(98, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(98),
        ])

        self.transform_screen = T.Compose([
            T.Resize((240, 320), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        ])

        self.transform1_g = T.Compose([
              T.Resize((84, 84), interpolation=T.InterpolationMode.BICUBIC),
        ])

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        self.dino = hubconf.dinov2_vits14()
        self.dino = self.dino.cuda()

        self.rnd = RNDModel(768, 768).cuda()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs
        self.ir = True
        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()

        self.obss = [None] * (shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

        # BELOW FOR DEBUGGING
        self.trajectory_vis = []
        self.global_episode = 0

        # CLUSTER TABLE variables and hyperparameters
        self.kappa = 0.8
        self.M = 250
        self.embedding_size = 384

        self.embs_memory = []
        self.obs_memory = []

        # Add two zero entries at the beggining for comparison
        # embedding, image (debug), lambda, appearances (debug)
        self.cluster_table = [[np.zeros(self.embedding_size), np.zeros((240, 320, 3)), 0, 0]]
        self.cluster_table = np.vstack((self.cluster_table, [np.zeros(self.embedding_size), np.zeros((240, 320, 3)), 0, 0]))

        self.gmm = GaussianMixture(n_components=self.M, reg_covar=1e-6, covariance_type="full", random_state=1, warm_start=False)


    def store_rnd_embs(self, obs):
        img_dino = self.transform1_g(torch.moveaxis(torch.from_numpy(np.stack(obs["screens"])), -1, 1)).float().cuda() / 255.
        embs = self.rnd.target(img_dino).detach().cpu().numpy()
        
        self.embs_memory.append(np.stack(embs))
        self.obs_memory.append(np.stack(obs["screens"]))

        return embs

    
    def store_dino_embs(self, obs):
        img_dino = self.transform_dino(torch.moveaxis(torch.from_numpy(np.stack(obs["screens"])), -1, 1)).float().cuda() / 255.
        embs = self.dino.forward_features(img_dino)["x_norm_clstoken"].detach().cpu().numpy()
        
        self.embs_memory.append(np.stack(embs))
        self.obs_memory.append(np.stack(obs["screens"]))

        return embs


    def update_cluster_table(self, embeddings, observations, rewards):
        cluster_dict = {}

        embeddings = np.stack(embeddings)
        embeddings = embeddings.reshape(embeddings.shape[0]*embeddings.shape[1], -1)
        observations = np.stack(observations)
        observations = observations.reshape(observations.shape[0]*observations.shape[1], 240, 320, 3)
        rewards = rewards.reshape(rewards.shape[0]*rewards.shape[1], -1)

        # Step 1 (Build episodic clusters) 
        # Fit GMM (Equation 1)
        self.gmm.fit(embeddings)
        m_prime = self.gmm.predict(embeddings)

        # Assign embeddings to episodic clusters
        for index, embedding in enumerate(embeddings):
            if m_prime[index] not in cluster_dict:
                cluster_dict[m_prime[index]] = []
            cluster_dict[m_prime[index]].append([observations[index], embedding, index])

		# Step 2 (Updating the global cluster table)
        for m in cluster_dict: 
            episodic_cluster = np.stack(cluster_dict[m])
            lambda_ep = episodic_cluster.shape[0]

			# Calculate episodic cluster center 
            mean_img = np.mean(np.stack(episodic_cluster[:, 0]), 0)
            mu_ep = np.mean(np.stack(episodic_cluster[:, 1]), 0)
            mu_globals = np.stack(self.cluster_table[:, 0])

            # Step 2.1 (Assigning episodic cluster centers to global cluster centers)
            distance = cosine_similarity(np.expand_dims(mu_ep, 0), mu_globals)
            k_prime = np.argmax(distance)
            d_prime = np.max(distance)

            # Step 2.2 (Calculating intrinsic rewards)				
            iverson_bracket_count = 0
            max_lambda_global = self.cluster_table[k_prime, 2]
            lambda_global = 0 if d_prime < self.kappa else max_lambda_global

            for index, _ in enumerate(embeddings): # Step 3 (Calculating intrinsic rewards)
                if m_prime[index] == m:
                    iverson_bracket_count += 1 
                    rho = lambda_global + iverson_bracket_count 
 
                    ir = 1 / np.sqrt(rho)
                    rewards[index] += 0.1*ir
            
			# Step 2.3 (Growing the cluster table)
            if d_prime < self.kappa: 
                self.cluster_table = np.vstack((self.cluster_table, [mu_ep, mean_img, lambda_ep, 1]))
            else:
				# Step 2.4 (Updating the global pseudo counts)
                self.cluster_table[k_prime, 2] += lambda_ep 
                self.cluster_table[k_prime, 3] += 1

        return rewards # Step 3


    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        print("ROLLOUT")
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            #print("CURRENT STEP: ", i)

            # Pass observations through pre-trained model
            if self.ir:
                #embs = self.store_dino_embs(self.obs)
                embs = self.store_rnd_embs(self.obs)
            
            self.trajectory_vis.append(np.concatenate((self.obs['automaps'][0], self.obs['screens'][0])))

            # Evaluate model and get next transition
            with torch.no_grad():
                if self.acmodel.recurrent:
                    img = self.transform_atari(torch.moveaxis(
                                                torch.from_numpy(np.stack(self.obs["screens"])), -1, 1)).float().cuda() / 255.
                    dist, value, memory = self.acmodel(img, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(self.obs)
            
            action = dist.sample()
            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            
            done = terminated or truncated
         
            # Update experiences values
            self.obss[i] = self.transform_atari(torch.moveaxis(
                                                torch.from_numpy(np.stack(self.obs["screens"])), -1, 1)).float().cuda() / 255.
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            self.rewards[i] = torch.tensor(reward, device=self.device) # add intrinsic reward
            self.log_probs[i] = dist.log_prob(action)

            # Update log values
            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask


        # Update Cluster Table
        print("UPDATING CLUSTER TABLE")
        if self.ir:
            self.rewards = self.update_cluster_table(self.embs_memory, self.obs_memory, self.rewards).reshape(self.num_frames_per_proc, 
                                                                                                    self.num_procs)
        # DEBUG cluster table
        if self.ir and not self.no_logging_wandb:
            if self.global_episode % 10 == 0:
                print("SAVING CLUSTER TABLE")
                
                sorted_ct = np.array(sorted(np.stack(self.cluster_table), key=lambda x: x[2], reverse=True))
                for i in range(len(self.cluster_table)):
                    images = wandb.Image(sorted_ct[i, 1], 
                                                     caption=str(sorted_ct[i, 2]) + "  app:" + str(sorted_ct[i,3])) 
                    wandb.log({"examples": images})

        # Add advantage and return to experiences
        img = self.transform_atari(torch.moveaxis(
                                    torch.from_numpy(np.stack(self.obs["screens"])), -1, 1)).float().cuda() / 255.
        preprocessed_obs = self.preprocess_obss(img, device=self.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0
            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.
        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]

        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = torch.stack(exps.obs)#DictList({"image": exps.obs})#self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        #DEBUG
        if not self.no_logging_wandb:
            wandb.log({"video": wandb.Video(np.moveaxis(np.stack(self.trajectory_vis), -1, 1), fps=4)})

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        self.embs_memory = []
        self.obs_memory = []
        self.trajectory_vis = []
        self.global_episode += 1

        return exps, logs

    @abstractmethod
    def update_parameters(self):
        pass
