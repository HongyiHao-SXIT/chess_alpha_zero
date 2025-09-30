import chess

class ChessGame:
    def __init__(self):
        self.board = chess.Board()

    def reset(self):
        """重置棋局"""
        self.board.reset()
        return self.board

    def step(self, move):
        """执行一步动作"""
        self.board.push(move)
        done = self.board.is_game_over()
        return self.board, done

    def legal_moves(self):
        """获取所有合法动作"""
        return list(self.board.legal_moves)

    def result(self):
        """返回胜负结果: 1=白胜, -1=黑胜, 0=和棋"""
        if self.board.is_checkmate():
            return -1 if self.board.turn else 1
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0
        return None
