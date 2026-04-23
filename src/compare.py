import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import os

from hamster_env import HamsterEnv
from q_learning import evaluate as ql_evaluate, get_state, get_q_values, MAX_STEPS
from dqn import evaluate as dqn_evaluate, QNetwork, load_model, device


# ── helper: smooth a noisy reward curve for plotting ─────────────────────────
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
        ("ql_logs_sparse.npy",  "Q-Learning (sparse)",  "blue",   0),
        ("ql_logs_shaped.npy",  "Q-Learning (shaped)",  "cornflowerblue", 0),
        ("dqn_logs_sparse.npy", "DQN (sparse)",         "red",    1),
        ("dqn_logs_shaped.npy", "DQN (shaped)",         "salmon", 1),
    ]

    for fname, label, color, ax_idx in configs:
        if not os.path.exists(fname):
            print(f"  Missing {fname} -- run q_learning.py and dqn.py first")
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
        ("ql_logs_sparse.npy",  "Q-Learning (sparse)",  "blue",   0),
        ("ql_logs_shaped.npy",  "Q-Learning (shaped)",  "cornflowerblue", 0),
        ("dqn_logs_sparse.npy", "DQN (sparse)",         "red",    1),
        ("dqn_logs_shaped.npy", "DQN (shaped)",         "salmon", 1),
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
        fname = f"q_table_{variant}.pkl"
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
        fname = f"dqn_{variant}.pth"
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
        fname = f"dqn_logs_{variant}.npy"
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


def print_qualitative_summary(results):
    """
    Write out a brief qualitative summary alongside the numbers (5 pts).
    I'll copy this into my README evaluation section.
    """
    if not results:
        return

    print("\n")
    print("=" * 65)
    print("QUALITATIVE SUMMARY")
    print("=" * 65)

    ql_sparse  = results.get("QL_sparse",  {})
    ql_shaped  = results.get("QL_shaped",  {})
    dqn_sparse = results.get("DQN_sparse", {})
    dqn_shaped = results.get("DQN_shaped", {})

    # compare algorithms
    if ql_sparse and dqn_sparse:
        ql_wr  = ql_sparse.get("win_rate_%", 0)
        dqn_wr = dqn_sparse.get("win_rate_%", 0)
        better = "DQN" if dqn_wr > ql_wr else "Q-Learning"
        diff   = abs(dqn_wr - ql_wr)
        print(f"\nAlgorithm comparison (sparse reward):")
        print(f"  {better} achieves a higher win rate by {diff:.1f} percentage points.")
        if diff < 5:
            print("  The gap is small, which makes sense -- 5x5 is simple enough")
            print("  for tabular Q-learning to handle well.")
        else:
            print("  DQN's neural network generalizes better across unseen states.")

    # compare reward shaping
    if ql_sparse and ql_shaped:
        diff = ql_shaped.get("win_rate_%", 0) - ql_sparse.get("win_rate_%", 0)
        print(f"\nReward shaping effect on Q-Learning: {diff:+.1f}% win rate")
        print("  Shaped reward adds a small bonus for moving toward seeds/magic.")
        if diff > 0:
            print("  This helps the agent explore more efficiently early in training.")
        else:
            print("  Surprisingly, shaping didn't help much -- the sparse reward")
            print("  signal was clear enough on this small map.")

    if dqn_sparse and dqn_shaped:
        diff = dqn_shaped.get("win_rate_%", 0) - dqn_sparse.get("win_rate_%", 0)
        print(f"\nReward shaping effect on DQN: {diff:+.1f}% win rate")

    print("\n(Full discussion in README.md Evaluation section)")
    print("=" * 65)


# main 
if __name__ == "__main__":
    print("=== Plotting learning curves ===")
    plot_learning_curves()

    print("\n=== Plotting win rates ===")
    plot_win_rates()

    print("\n=== Plotting DQN loss ===")
    plot_loss_curve()

    print("\n=== Running ablation study ===")
    results = run_ablation()

    print("\n=== Qualitative summary ===")
    print_qualitative_summary(results)

    print("\nAll done! Generated files:")
    print("  learning_curves.png")
    print("  win_rates.png")
    print("  dqn_loss.png")
    print("  ablation_results.npy")