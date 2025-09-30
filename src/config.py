class Config:
    board_size = 8
    n_simulations = 100       # MCTS 模拟次数
    c_puct = 1.0              # MCTS 探索参数
    buffer_size = 10000
    batch_size = 64
    lr = 1e-3
    n_epochs = 10
    device = "cuda"           # 如果有 GPU, 否则改成 "cpu"
