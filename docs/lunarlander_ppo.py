"""
Sanity check on PPO with LunarLander-v3
with five seeds
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

SEEDS = [12, 34, 56, 78, 910]
SOLVED = 200

def load_logs():
    rewards, steps, counts = [], [], None
    for seed in SEEDS:
        df = pd.read_csv(f"logs/gym_lunarlander_ppo_seed{seed}.csv")

        if counts is None:
            counts = df["count"].to_numpy()
        
        rewards.append(df["ep_reward_average"].to_numpy())
        steps.append(df["ep_step_average"].to_numpy())

    return rewards, steps, counts

def plot(data, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax_curve = plt.subplots(1, 1, figsize=(12, 5))

    rewards, steps, counts = data

    median = np.median(rewards, axis=0)
    q25 = np.percentile(rewards, 25, axis=0)
    q75 = np.percentile(rewards, 75, axis=0)

    (line, ) = ax_curve.plot(counts, median, label="PPO", linewidth=2)
    ax_curve.fill_between(counts, q25, q75, alpha=0.15, color=line.get_color())

    ax_curve.axhline(SOLVED, ls="--", lw=1.2, color="gray")
    ax_curve.text(counts[-1] * 0.02, SOLVED + 8, f"solved ({SOLVED:.0f})", ha="left", va="bottom", fontsize=9, color="gray")

    ax_curve.set_xlabel("Environment steps")
    ax_curve.set_ylabel("Episode return (100-ep moving average)")
    ax_curve.set_title("LunarLander-v3: Learning Curves")
    ax_curve.set_ylim(-300, 300)
    ax_curve.set_xlim(0, counts[-1])
    ax_curve.xaxis.set_major_formatter(
        ticker.StrMethodFormatter("{x:,.0f}")
    )
    ax_curve.legend(loc="lower right", framealpha=0.9)

    fig.suptitle("median over 5 seeds; shaded region: interquartile range", y=0.005, fontsize=9, color="gray")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def main():
    data = load_logs()
    plot(data, "docs/ppo_plots")

if __name__ == "__main__":
    main()