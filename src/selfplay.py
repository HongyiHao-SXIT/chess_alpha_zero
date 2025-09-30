# src/selfplay.py
"""
Parallel (and single-process) self-play data generation.

- parallel_generate_selfplay_games(...) launches multiple processes (multiprocessing.Pool)
  Each worker loads the model weights from a temporary file models/_tmp_worker.pth and
  runs its assigned number of games, returning samples.

- generate_selfplay_games(...) is the existing single-process generator (kept for debugging).
"""

import os
import tempfile
import math
import shutil
import multiprocessing as mp
from typing import List, Tuple
import numpy as np
import chess
import torch

# imports from local package (assumes this file resides in src/)
from model import PolicyValueNet
from mcts import MCTS
from chess_env import board_to_tensor, ACTION_SIZE, index_to_move
from config import Config

# ---- Single-process generator (unchanged, useful for debugging) ----
def generate_selfplay_games(net: PolicyValueNet,
                            num_games: int = 1,
                            n_simulations: int = None,
                            device: str = None,
                            verbose: bool = False) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """Run self-play serially, return list of (state, policy, z)."""
    if device is None:
        device = Config.device
    if n_simulations is None:
        n_simulations = Config.n_simulations

    net.to(device)
    net.eval()
    mcts = MCTS(net, n_simulations=n_simulations, c_puct=Config.c_puct, device=device)

    all_samples = []

    for _ in range(num_games):
        board = chess.Board()
        states, policies, players = [], [], []
        while not board.is_game_over():
            probs = mcts.run(board)
            if not probs:
                break
            pi = np.zeros(ACTION_SIZE, dtype=np.float32)
            actions = list(probs.keys())
            probs_arr = np.array(list(probs.values()), dtype=np.float32)
            s = probs_arr.sum()
            if s > 0:
                probs_arr = probs_arr / s
            else:
                probs_arr = np.ones_like(probs_arr) / len(probs_arr)
            for a, p in zip(actions, probs_arr):
                pi[a] = p
            states.append(board_to_tensor(board))
            policies.append(pi)
            players.append(board.turn)
            action = np.random.choice(actions, p=probs_arr)
            mv = index_to_move(action, board)
            if mv is None:
                # fallback uniform legal move
                legal = list(board.legal_moves)
                mv = np.random.choice(legal)
            board.push(mv)

        # finalize game result
        res_str = board.result()
        if res_str == "1-0":
            final = 1.0
        elif res_str == "0-1":
            final = -1.0
        else:
            final = 0.0

        for s, p, player in zip(states, policies, players):
            z = final if player else -final
            all_samples.append((s, p, z))

    return all_samples


# ---- Worker function for multiprocessing ----
# This must be a top-level function to be picklable by multiprocessing.
def _worker_selfplay(task):
    """
    Worker entrypoint for Pool.
    task: dict with keys:
      - model_path: path to state_dict file
      - num_games: int
      - n_simulations: int
      - device: str (cpu recommended for workers)
    returns: list of samples (s,p,z)
    """
    model_path = task["model_path"]
    num_games = task["num_games"]
    n_simulations = task.get("n_simulations", Config.n_simulations)
    device = task.get("device", "cpu")

    # Local imports inside worker (helps Windows spawn behavior)
    from model import PolicyValueNet
    from mcts import MCTS
    from chess_env import board_to_tensor, ACTION_SIZE, index_to_move
    import chess
    import numpy as np
    import torch

    # load model in worker
    net = PolicyValueNet()
    try:
        net.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        # if model load fails, return empty list
        print(f"[worker] Failed to load model from {model_path}: {e}")
        return []

    net.to(device)
    net.eval()
    mcts = MCTS(net, n_simulations=n_simulations, c_puct=Config.c_puct, device=device)

    samples = []
    for _ in range(num_games):
        board = chess.Board()
        states, policies, players = [], [], []
        while not board.is_game_over():
            probs = mcts.run(board)
            if not probs:
                break
            pi = np.zeros(ACTION_SIZE, dtype=np.float32)
            actions = list(probs.keys())
            probs_arr = np.array(list(probs.values()), dtype=np.float32)
            ssum = probs_arr.sum()
            if ssum > 0:
                probs_arr = probs_arr / ssum
            else:
                probs_arr = np.ones_like(probs_arr) / len(probs_arr)
            for a, p in zip(actions, probs_arr):
                pi[a] = p
            states.append(board_to_tensor(board))
            policies.append(pi)
            players.append(board.turn)
            action = np.random.choice(actions, p=probs_arr)
            mv = index_to_move(action, board)
            if mv is None:
                legal = list(board.legal_moves)
                mv = np.random.choice(legal)
            board.push(mv)

        # finalize
        res_str = board.result()
        if res_str == "1-0":
            final = 1.0
        elif res_str == "0-1":
            final = -1.0
        else:
            final = 0.0

        for s, p, player in zip(states, policies, players):
            z = final if player else -final
            samples.append((s, p, z))

    return samples


# ---- Parallel entrypoint ----
def parallel_generate_selfplay_games(net: PolicyValueNet,
                                    num_games: int = 8,
                                    num_workers: int = None,
                                    n_simulations: int = None,
                                    worker_device: str = "cpu",
                                    tmp_model_path: str = None) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Parallel self-play:
      - Saves net.state_dict() to a temporary file (tmp_model_path or models/_tmp_worker.pth)
      - Spawns a Pool of workers, each loads that model and runs its share of games
      - Returns combined list of samples

    Args:
      net: PolicyValueNet (in main process)
      num_games: total games to generate
      num_workers: number of processes to spawn (default = cpu_count())
      n_simulations: MCTS sims per move (defaults to Config)
      worker_device: device for workers (recommend 'cpu')
      tmp_model_path: optional explicit temp path to write weights
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    if n_simulations is None:
        n_simulations = Config.n_simulations

    # Ensure models dir exists
    os.makedirs("models", exist_ok=True)

    # save model state_dict to temp file for workers to load
    if tmp_model_path is None:
        tmp_model_path = os.path.join("models", "_tmp_worker.pth")
    # save on CPU to maximize compatibility
    torch.save(net.state_dict(), tmp_model_path)

    # distribute games across workers
    base = num_games // num_workers
    extras = num_games % num_workers
    tasks = []
    for i in range(num_workers):
        g = base + (1 if i < extras else 0)
        if g <= 0:
            continue
        task = {
            "model_path": tmp_model_path,
            "num_games": g,
            "n_simulations": n_simulations,
            "device": worker_device
        }
        tasks.append(task)

    if len(tasks) == 0:
        return []

    # Use multiprocessing Pool
    # Windows requires 'spawn' start method; ensure it's available
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        results = pool.map(_worker_selfplay, tasks)

    # combine results (flatten)
    all_samples = []
    for r in results:
        if r:
            all_samples.extend(r)

    # remove temp model file if desired
    try:
        os.remove(tmp_model_path)
    except Exception:
        pass

    return all_samples
