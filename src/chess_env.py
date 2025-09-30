# src/chess_env.py
import numpy as np
import chess

ACTION_SIZE = 64 * 64  # from_square * 64 + to_square = 4096


def board_to_tensor(board: chess.Board):
    """
    把 python-chess 的 board 转为 (12,8,8) float32 tensor（numpy）
    12 通道: 白 P N B R Q K, 黑 p n b r q k
    """
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = board.piece_map()
    for sq, piece in piece_map.items():
        r = 7 - (sq // 8)  # 将 0-63 映射到棋盘行：python-chess 默认 a1=0，a1 在底行 -> 我们把 tensor 第0行设为 top
        c = sq % 8
        symbol = piece.symbol()
        order = "PNBRQKpnbrqk"
        idx = order.index(symbol)
        planes[idx, r, c] = 1.0
    # 增加行动方通道（可选），但现在不做单独通道，网络可从 piece color 推断
    return planes  # shape (12,8,8)


def move_to_index(move: chess.Move):
    """
    将 chess.Move 转为 0..4095 的索引 (from_square*64 + to_square)
    注意: 升变没有额外编码（promotion 会被映射到相同的 from->to）。
    """
    return move.from_square * 64 + move.to_square


def index_to_move(idx: int, board: chess.Board):
    """
    将索引解码为 chess.Move（如果该 move 非法则返回 None）
    注意：如果该 from->to 是一个 promotion，必须手工尝试 promotion 类型以获得合法 move。
    这里先尝试无 promotion 的 move；如果不合法且存在 promotion，则尝试 queen promotion。
    """
    from_sq = idx // 64
    to_sq = idx % 64
    # 普通走子尝试
    mv = chess.Move(from_sq, to_sq)
    if mv in board.legal_moves:
        return mv
    # 如果非法，尝试 promotion（尝试 queen, rook, bishop, knight）
    for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
        mvp = chess.Move(from_sq, to_sq, promotion=promo)
        if mvp in board.legal_moves:
            return mvp
    return None


class ChessEnv:
    """
    简单的 Gym 风格封装（非依赖 gym 库）
    state: board_to_tensor(board)
    action: int in [0, ACTION_SIZE)
    """

    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        self.board.reset()
        return board_to_tensor(self.board)

    def legal_actions(self):
        return [move_to_index(mv) for mv in self.board.legal_moves]

    def step(self, action_idx: int):
        mv = index_to_move(action_idx, self.board)
        if mv is None:
            # 非法走子：视为非法，并不改变局面；你可以改为抛异常或随机选合法走子
            # 这里我们选择抛异常，调用者（MCTS/selfplay）应保证选的是合法动作
            raise ValueError(f"Illegal action idx {action_idx} for board:\n{self.board}")
        self.board.push(mv)
        done = self.board.is_game_over()
        reward = None
        if done:
            # result(): '1-0', '0-1', '1/2-1/2'
            res = self.board.result()
            if res == "1-0":
                reward = 1.0
            elif res == "0-1":
                reward = -1.0
            else:
                reward = 0.0
        return board_to_tensor(self.board), reward, done

    def get_board(self):
        return self.board

    def set_board(self, board: chess.Board):
        self.board = board
