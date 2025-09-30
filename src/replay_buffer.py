# src/replay_buffer.py
import os
import random
import pickle
from collections import deque
import numpy as np
import sys, pathlib
# ensure imports work when run from project root
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, policy, z):
        """
        state: numpy array (12,8,8)
        policy: numpy array (ACTION_SIZE,) probability distribution
        z: float in {-1,0,1} from current player's viewpoint
        """
        self.buffer.append((state.astype('float32'), policy.astype('float32'), float(z)))

    def extend(self, samples):
        for s, p, z in samples:
            self.push(s, p, z)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, k=min(batch_size, len(self.buffer)))
        states = np.stack([b[0] for b in batch], axis=0)
        policies = np.stack([b[1] for b in batch], axis=0)
        zs = np.array([b[2] for b in batch], dtype='float32')
        return states, policies, zs

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.buffer = deque(data, maxlen=self.capacity)
