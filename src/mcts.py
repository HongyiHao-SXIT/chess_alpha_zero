# src/mcts.py
import math
import numpy as np
import torch
from collections import defaultdict
from copy import deepcopy
import chess

from chess_env import move_to_index, index_to_move, board_to_tensor, ACTION_SIZE
# 注意：ACTION_SIZE 与 model.ACTION_SIZE 应保持一致

class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, prior=0.0):
        self.board = board  # python-chess Board (deepcopy 外部保证)
        self.parent = parent
        self.prior = prior  # P(s,a)
        self.children = {}  # action_idx -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0

    def q_value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


class MCTS:
    def __init__(self, net, n_simulations=100, c_puct=1.0, device="cpu"):
        """
        net: PolicyValueNet (torch) in eval mode
        """
        self.net = net
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.device = device

    def run(self, root_board: chess.Board):
        """
        对给定 board 运行 MCTS，返回根节点 action_probs: dict action_idx -> prob
        """
        root = MCTSNode(deepcopy(root_board), parent=None, prior=1.0)

        # If terminal at root, return trivially
        if root.board.is_game_over():
            return {}

        # Expand root with network prior
        self._expand(root)

        for _ in range(self.n_simulations):
            node = root
            search_path = [node]
            # selection
            while node.children:
                action, node = self._select(node)
                search_path.append(node)

            # now node is a leaf
            value = self._evaluate(node)

            # backup
            self._backup(search_path, value)

        # compute action probabilities from visit counts
        counts = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)
        if counts.sum() == 0:
            # fallback uniform
            probs = {}
            for a in root.children.keys():
                probs[a] = 1.0 / len(root.children)
            return probs

        probs_arr = counts / counts.sum()
        probs = {}
        for a, p in zip(root.children.keys(), probs_arr):
            probs[a] = float(p)
        return probs

    def _select(self, node: MCTSNode):
        """
        在 node 的子节点中选择具有最大 UCB 的动作
        """
        total_N = sum(child.visit_count for child in node.children.values())
        best_score = -float("inf")
        best_action = None
        best_child = None
        for action, child in node.children.items():
            Q = child.q_value()
            U = self.c_puct * child.prior * math.sqrt(total_N + 1e-8) / (1 + child.visit_count)
            score = Q + U
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def _evaluate(self, node: MCTSNode):
        """
        对当前叶子节点进行评估：
        - 如果终局，直接根据结果返回 value
        - 否则，用网络预测 (policy, value)，并扩展节点（children）
        返回 value (从当前玩家视角，+1 表示当前玩家最终会赢)
        """
        if node.board.is_game_over():
            result = node.board.result()
            if result == "1-0":
                # white won
                return 1.0 if node.board.turn == chess.BLACK else -1.0
            elif result == "0-1":
                return 1.0 if node.board.turn == chess.WHITE else -1.0
            else:
                return 0.0

        # use network to get prior and value
        tensor = board_to_tensor(node.board)  # (12,8,8) numpy
        x = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1,12,8,8)
        with torch.no_grad():
            logits, value = self.net(x)  # logits shape (1, ACTION_SIZE), value shape (1,)
            logits = logits.squeeze(0).cpu().numpy()
            value = float(value.item())

        # convert logits -> probabilities but only for legal moves
        legal = list(node.board.legal_moves)
        priors = {}
        ps = []
        acts = []
        for mv in legal:
            idx = move_to_index(mv)
            acts.append(idx)
            ps.append(math.exp(logits[idx]))  # logits -> unnormalized prob
        ps_sum = sum(ps)
        if ps_sum <= 0:
            # fallback uniform
            norm = 1.0 / len(ps)
            for a in acts:
                priors[a] = norm
        else:
            for a, p in zip(acts, ps):
                priors[a] = p / ps_sum

        # expand node
        for a, p in priors.items():
            # create child node with resulting board state
            new_board = deepcopy(node.board)
            mv = index_to_move(a, new_board)
            if mv is None:
                continue
            new_board.push(mv)
            node.children[a] = MCTSNode(new_board, parent=node, prior=p)

        return value

    def _expand(self, node: MCTSNode):
        """
        初始扩展（和 _evaluate 中做的一样），但保留单独调用
        """
        if node.board.is_game_over():
            return
        tensor = board_to_tensor(node.board)
        x = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(x)
            logits = logits.squeeze(0).cpu().numpy()

        legal = list(node.board.legal_moves)
        ps = []
        acts = []
        for mv in legal:
            idx = move_to_index(mv)
            acts.append(idx)
            ps.append(math.exp(logits[idx]))
        ps_sum = sum(ps)
        if ps_sum <= 0:
            for a in acts:
                node.children[a] = MCTSNode(deepcopy(node.board), parent=node, prior=1.0 / len(acts))
        else:
            for a, p in zip(acts, ps):
                node.children[a] = MCTSNode(deepcopy(node.board), parent=node, prior=p / ps_sum)

    def _backup(self, path, value):
        """
        回溯：value 从叶子视角返回（对手视角取反）
        path: list of nodes from root ... leaf
        """
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # 交替视角
