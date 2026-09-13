"""
Sanity check on DQN and NoisyNet-DQN with Rust native CartPole
Checked with five seeds
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

SEEDS = [12, 19, 45, 66, 80]
SOLVED = 475
VARIANT = ["dqn", "noisy_dqn"]
NAMES = {"dqn": "DQN", "noisy_dqn": "NoisyDQN"}

def load_logs(variant):
    rewards, steps, counts = [], [], None
    for seed in SEEDS:
        df = pd.read_csv(f"./logs/{variant}_test/{seed}.csv")

        if counts is None:
            counts = df["count"].to_numpy()

        rewards.append(df["ep_reward_average"].fillna(0.0).to_numpy())
        steps.append(df["ep_step_average"].fillna(0.0).to_numpy())

    return np.array(rewards), np.array(steps), counts

def plot_main(data, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(9, 5))
    counts = []
    for variant in VARIANT:
        rewards, steps, counts = data[variant]
        median = np.median(rewards, axis=0)
        q25 = np.quantile(rewards, 0.25, axis=0)
        q75 = np.quantile(rewards, 0.75, axis=0)

        (line, ) = ax.plot(counts, median, label=NAMES[variant], linewidth=2)
        ax.fill_between(counts, q25, q75, alpha=0.15, color=line.get_color())

    ax.axhline(SOLVED, ls="--", lw=1.2, color="gray")
    ax.text(counts[-1] * 0.02, SOLVED + 8, f"solved ({SOLVED:.0f})", ha="left", va="bottom", fontsize=9, color="gray")

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return (100-ep moving average)")
    ax.set_title("CartPole-v1: learning curve")
    ax.set_ylim(-10, 600)
    ax.set_xlim(0, counts[-1])
    ax.legend(loc="lower right", framealpha=0.9)

    fig.suptitle("median over 5 seeds; shaded region: interquartile range", y=0.005, fontsize=9, color="gray")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig

def main():
    data = {}
    for variant in VARIANT:
        data[variant] = load_logs(variant)
    plot_main(data, "docs/figures/cartpole")

if __name__ == "__main__":
    main()