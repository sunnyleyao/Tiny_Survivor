import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import os
import sys

ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA  = os.path.join(ROOT, "data")
MODEL = os.path.join(ROOT, "models")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hamster_env import HamsterEnv
from q_learning import evaluate as ql_evaluate, get_state, get_q_values, MAX_STEPS
from dqn import evaluate as dqn_evaluate, QNetwork, load_model, device


# smooth a noisy reward curve for plotting 
def smooth(values, window=500):
    """Running average over a window -- makes the learning curve readable."""
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window)
        smoothed.append(np.mean(values[start:i+1]))
    return smoothed


def plot_learning_curves():
    """
    Load all 4 training logs and plot reward over episodes.
    This shows convergence (3 pts) and lets me visually compare
    how fast each method learns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Learning Curves: Q-Learning vs DQN", fontsize=14)

    configs = [
        (os.path.join(DATA, "ql_logs_sparse.npy"),  "Q-Learning (sparse)",  "blue",   0),
        (os.path.join(DATA, "ql_logs_shaped.npy"),  "Q-Learning (shaped)",  "cornflowerblue", 0),
        (os.path.join(DATA, "dqn_logs_sparse.npy"), "DQN (sparse)",         "red",    1),
        (os.path.join(DATA, "dqn_logs_shaped.npy"), "DQN (shaped)",         "salmon", 1),
    ]

    for fname, label, color, ax_idx in configs:
        if not os.path.exists(fname):
            print(f"  Missing {fname}")
            continue
        logs = np.load(fname, allow_pickle=True).item()
        smoothed = smooth(logs["reward"], window=500)
        axes[ax_idx].plot(smoothed, label=label, color=color)

    axes[0].set_title("Q-Learning")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Avg Reward (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("DQN")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Avg Reward (smoothed)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150)
    plt.close()
    print("Saved learning_curves.png")


# plot win rate over training
def plot_win_rates():
    """
    Win rate over time -- a more intuitive metric than raw reward.
    Helps me see when the agent actually starts solving the task.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Win Rate Over Training", fontsize=14)

    configs = [
        (os.path.join(DATA, "ql_logs_sparse.npy"),  "Q-Learning (sparse)",  "blue",   0),
        (os.path.join(DATA, "ql_logs_shaped.npy"),  "Q-Learning (shaped)",  "cornflowerblue", 0),
        (os.path.join(DATA, "dqn_logs_sparse.npy"), "DQN (sparse)","red", 1),
        (os.path.join(DATA, "dqn_logs_shaped.npy"), "DQN (shaped)","salmon", 1),
    ]

    for fname, label, color, ax_idx in configs:
        if not os.path.exists(fname):
            continue
        logs     = np.load(fname, allow_pickle=True).item()
        smoothed = smooth(logs["wins"], window=500)
        pct      = [v * 100 for v in smoothed]
        axes[ax_idx].plot(pct, label=label, color=color)

    for ax, title in zip(axes, ["Q-Learning", "DQN"]):
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Win Rate % (smoothed)")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("win_rates.png", dpi=150)
    plt.close()
    print("Saved win_rates.png")


# ablation table 
def run_ablation():
    obs_dim = HamsterEnv().observation_space.shape[0]
    results = {}

    # Q-Learning variants
    for variant in ["sparse", "shaped"]:
        fname = os.path.join(MODEL, f"q_table_{variant}.pkl")
        if not os.path.exists(fname):
            print(f"  Missing {fname}, skipping...")
            continue
        with open(fname, "rb") as f:
            q_table = pickle.load(f)
        shaped = (variant == "shaped")
        print(f"\nEvaluating Q-Learning ({variant})...")
        results[f"QL_{variant}"] = ql_evaluate(q_table, shaped_reward=shaped)

    # DQN variants
    for variant in ["sparse", "shaped"]:
        fname = os.path.join(MODEL, f"dqn_{variant}.pth")
        if not os.path.exists(fname):
            print(f"  Missing {fname}, skipping...")
            continue
        model  = load_model(fname, obs_dim)
        shaped = (variant == "shaped")
        print(f"\nEvaluating DQN ({variant})...")
        results[f"DQN_{variant}"] = dqn_evaluate(model, shaped_reward=shaped)

    if not results:
        print("No results to show -- train the agents first.")
        return results

    # print the 2x2 ablation table
    print("\n")
    print("=" * 65)
    print("ABLATION STUDY -- 2x2 Summary Table")
    print("Rows: Algorithm | Columns: Reward Function")
    print("=" * 65)
    print(f"{'':22} | {'Sparse Reward':>18} | {'Shaped Reward':>18}")
    print("-" * 65)

    metrics = ["avg_reward", "win_rate_%", "avg_steps"]

    for algo in ["QL", "DQN"]:
        label = "Q-Learning" if algo == "QL" else "DQN"
        for metric in metrics:
            sparse_val = results.get(f"{algo}_sparse", {}).get(metric, "N/A")
            shaped_val = results.get(f"{algo}_shaped", {}).get(metric, "N/A")
            if metric == "avg_reward":
                row_label = f"  {label} -- avg reward"
            elif metric == "win_rate_%":
                row_label = f"  {label} -- win rate %"
            else:
                row_label = f"  {label} -- avg steps"
            sv = f"{sparse_val:.2f}" if isinstance(sparse_val, float) else str(sparse_val)
            shv = f"{shaped_val:.2f}" if isinstance(shaped_val, float) else str(shaped_val)
            print(f"{row_label:<22} | {sv:>18} | {shv:>18}")
        print("-" * 65)

    print("=" * 65)

    np.save("ablation_results.npy", results)
    print("\nSaved ablation_results.npy")

    return results


def plot_loss_curve():
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("DQN Training Loss")

    for variant, color in [("sparse", "red"), ("shaped", "salmon")]:
        fname = os.path.join(DATA, f"dqn_logs_{variant}.npy")
        if not os.path.exists(fname):
            continue
        logs     = np.load(fname, allow_pickle=True).item()
        smoothed = smooth(logs["loss"], window=200)
        ax.plot(smoothed, label=f"DQN ({variant})", color=color)

    ax.set_xlabel("Episode")
    ax.set_ylabel("MSE Loss (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("dqn_loss.png", dpi=150)
    plt.close()
    print("Saved dqn_loss.png")


if __name__ == "__main__":
    plot_learning_curves()
    plot_win_rates()
    plot_loss_curve()
    results = run_ablation()