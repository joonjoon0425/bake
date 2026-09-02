"""
Compare DQN-like algorithms (DQN, double DQN, DQN with PER, dueling DQN, NoisyNet-DQN)
with five seeds
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

SEEDS = ["12", "34", "56", "78", "910"]
VARIANTS = ["dqn", "ddqn", "dqn_per", "dueling_dqn", "noisy_dqn"]
TAGS = {"dqn": "DQN", "ddqn": "Double DQN", "dqn_per": "DQN with PER", "dueling_dqn": "Dueling DQN", "noisy_dqn": "NoisyNet-DQN"}

GAMMA = 0.99
Q_BOUND = 1.0 / (1.0 - GAMMA)
SOLVED = 475.0

def load_logs(variant):
    rewards, qmeans, steps = [], [], None

    for seed in SEEDS:
        path = f"logs/{variant}_seed{seed}.csv"

        df = pd.read_csv(path)

        if steps is None:
            steps = df["count"].to_numpy()

        rewards.append(df["ep_reward_average"].fillna(0.0).to_numpy())
        qmeans.append(df["qmean"].to_numpy())

    return steps, np.array(rewards), np.array(qmeans)

def plot_main(data, out_path):
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax_curve = plt.subplots(1, 1, figsize=(12, 5))
    steps = []
    for variant in VARIANTS:
        steps, rewards, _ = data[variant]

        median = np.median(rewards, axis=0)
        q25 = np.percentile(rewards, 25, axis=0)
        q75 = np.percentile(rewards, 75, axis=0)

        (line, ) = ax_curve.plot(steps, median, label=TAGS[variant], linewidth=2)
        ax_curve.fill_between(steps, q25, q75, alpha=0.15, color=line.get_color())

    ax_curve.axhline(SOLVED, ls="--", lw=1.2, color="gray")

    ax_curve.text(steps[-1] * 0.02, SOLVED + 8, f"solved ({SOLVED:.0f})", ha="left", va="bottom", fontsize=9, color="gray")

    ax_curve.set_xlabel("Environment steps")
    ax_curve.set_ylabel("Episode return (100-ep moving average)")
    ax_curve.set_title("CartPole-v1: Learning Curves")
    ax_curve.set_ylim(0, 520)
    ax_curve.legend(loc="lower right", framealpha=0.9)

    fig.suptitle("median over 5 seeds; shaded region: interquartile range", y=0.005, fontsize=9, color="dimgray")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig

def table(data):
    rows = []
    for variant in VARIANTS:
        steps, rewards, qmeans = data[variant]

        reach = []
        for reward in rewards:
            idx = np.argmax(reward >= 475)
            reach.append(steps[idx] if reward[idx] >= 475 else np.nan)
        reach = np.array(reach, dtype=float)
        n_solved = int(np.sum(~np.isnan(reach)))
        reach_median = np.nanmedian(reach) if n_solved else np.nan
        final = np.median(rewards[:, -10:].mean(axis=1))

        q_max = np.median(qmeans.max(axis=1))
        rows.append({
            "algorithm": TAGS[variant],
            "steps to 475": "-" if np.isnan(reach_median) else f"{reach_median / 1000:.0f}k",
            "solved seeds": f"{n_solved} / 5",
            "final return": f"{final:.1f}",
            "maximum q mean": f"{q_max:.1f}"
        })

    df = pd.DataFrame(rows)

    print("\n" + df.to_string(index=False))
    print("\n--- 마크다운 ---\n")
    print(df.to_markdown(index=False))
    return df


def main():
    # load data
    data = {}
    for variant in VARIANTS:
        data[variant] = load_logs(variant)
    
    plot_main(data, "docs/dqn_plots.png")
    table(data)

if __name__ == "__main__":
    main()