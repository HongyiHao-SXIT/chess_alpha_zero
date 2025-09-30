# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_SIZE = 64 * 64  # 4096

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class PolicyValueNet(nn.Module):
    def __init__(self, in_channels=12, num_blocks=6, channels=128):
        """
        in_channels: 12 planes
        num_blocks: 残差块数量（可调）
        channels: 卷积通道数
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

        # residual blocks
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, ACTION_SIZE)

        # value head
        self.value_conv = nn.Conv2d(channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        x: tensor shape (B, 12, 8, 8)
        returns:
          policy_logits: (B, ACTION_SIZE)
          value: (B,) in [-1,1]
        """
        if x.dtype != torch.float32:
            x = x.float()
        x = F.relu(self.bn(self.conv(x)))
        x = self.blocks(x)

        # policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # logits (no softmax)

        # value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return p, v


def load_model(path, device="cpu"):
    net = PolicyValueNet()
    net.load_state_dict(torch.load(path, map_location=device))
    net.to(device)
    net.eval()
    return net
