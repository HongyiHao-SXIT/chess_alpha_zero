# src/main.py
import os, sys, pathlib
from tqdm import trange

# allow running as: python src/main.py (project root)
sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from train import Trainer
from selfplay import generate_selfplay_games, parallel_generate_selfplay_games
from config import Config


def ensure_dirs():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)


def main():
    ensure_dirs()
    trainer = Trainer(device=Config.device)

    # if there's an existing model, try load it (optional)
    model_file = os.path.join("models", "pv_net.pth")
    if os.path.exists(model_file):
        print("Loading existing model:", model_file)
        trainer.load(model_file)

    n_iterations = 50   # 外层循环次数（自博弈+训练）
    games_per_iter = 2  # 每轮自博弈局数（训练时建议更大）
    train_steps = 100   # 每轮训练步数
    batch_size = Config.batch_size

    for it in trange(n_iterations, desc="Iteration"):
        print(f"\n=== Iteration {it+1}/{n_iterations} ===")

        # 1) Self-play (单进程版本，调试时可用)
        #samples = generate_selfplay_games(trainer.net, num_games=games_per_iter, verbose=True)
        #print(f"Generated {len(samples)} training samples from {games_per_iter} games")
        #trainer.add_selfplay_samples(samples)

        # 2) Train
        stats = trainer.train_epochs(epochs=1, steps_per_epoch=train_steps)
        if stats:
            avg_loss = sum(s["loss"] for s in stats) / len(stats)
            print(f"Train avg loss: {avg_loss:.4f}")

        # 3) Save model
        trainer.save()
        print(f"Saved model to {trainer.model_path}")

    print("Training loop finished.")

    num_games = 8
    num_workers = 4  # 或 None 使用 cpu_count()-1
    samples = parallel_generate_selfplay_games(trainer.net,
                                               num_games=num_games,
                                               num_workers=num_workers,
                                               n_simulations=Config.n_simulations,
                                               worker_device="cpu")
    print(f"Generated {len(samples)} samples from {num_games} games using {num_workers} workers")
    trainer.add_selfplay_samples(samples)


if __name__ == "__main__":
    main()
