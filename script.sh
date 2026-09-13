printf '%s\n' 80 12 45 66 19 | xargs -P 5 -I{} sh -c \
    ' ./target/release/examples/dueling_dqn_lunarlander {} > "logs/dueling_dqn_lunarlander/{}.csv" '