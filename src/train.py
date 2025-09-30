# src/train.py
import os, sys, pathlib, time
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import PolicyValueNet
from replay_buffer import ReplayBuffer
from config import Config
from tqdm import trange

class Trainer:
    def __init__(self, device=None, model_path=None):
        self.device = device or Config.device
        self.net = PolicyValueNet().to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=Config.lr, weight_decay=1e-4)
        self.buffer = ReplayBuffer(capacity=Config.buffer_size)
        self.model_path = model_path or os.path.join("models", "pv_net.pth")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))

    def save(self, path=None):
        path = path or self.model_path
        torch.save(self.net.state_dict(), path)

    def add_selfplay_samples(self, samples):
        self.buffer.extend(samples)

    def train_step(self, batch_size=Config.batch_size):
        if len(self.buffer) == 0:
            return None

        states_np, policies_np, zs_np = self.buffer.sample(batch_size)
        # to tensors
        states = torch.tensor(states_np, dtype=torch.float32).to(self.device)  # (B,12,8,8)
        policies = torch.tensor(policies_np, dtype=torch.float32).to(self.device)  # (B,A)
        zs = torch.tensor(zs_np, dtype=torch.float32).to(self.device)  # (B,)

        self.net.train()
        logits, values = self.net(states)  # logits (B,A), values (B,)
        # value loss (MSE)
        value_loss = torch.mean((values - zs) ** 2)
        # policy loss: cross entropy between target distribution (policies) and model logits
        log_probs = torch.log_softmax(logits, dim=1)
        policy_loss = -torch.mean(torch.sum(policies * log_probs, dim=1))
        loss = value_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "value_loss": float(value_loss.item()),
            "policy_loss": float(policy_loss.item())
        }

    def train_epochs(self, epochs=1, steps_per_epoch=100):
        stats = []
        for e in range(epochs):
            for _ in trange(steps_per_epoch, desc=f"Train epoch {e+1}/{epochs}"):
                res = self.train_step()
                if res is not None:
                    stats.append(res)
        return stats
