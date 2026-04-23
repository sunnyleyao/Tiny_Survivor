

import numpy as np
import pickle
from hamster_env import HamsterEnv

EPISODES  = 50000
LR        = 0.5     # higher LR for faster convergence
GAMMA     = 0.95
EPS_START = 1.0
EPS_END   = 0.05
EPS_DECAY = 0.9998
MAX_STEPS = 200

def get_state(obs, grid_size=5):
    n = grid_size * grid_size

    row = int(round(obs[0] * (grid_size - 1)))
    col = int(round(obs[1] * (grid_size - 1)))

    seed_map  = obs[2 : 2 + n].reshape(grid_size, grid_size)
    magic_map = obs[2 + n: 2 + 2*n].reshape(grid_size, grid_size)
    trap_map  = obs[2 + 2*n: 2 + 3*n].reshape(grid_size, grid_size)

    def nearest_dir(item_map, cur_r, cur_c):
        """Return direction to nearest item as a simple code."""
        best_d = float("inf")
        best_r, best_c = -1, -1
        for r in range(grid_size):
            for c in range(grid_size):
                if item_map[r, c] > 0.5:
                    d = abs(r - cur_r) + abs(c - cur_c)
                    if d < best_d:
                        best_d = d
                        best_r, best_c = r, c
        if best_r == -1:
            return 4   # no item found
        dr = best_r - cur_r
        dc = best_c - cur_c
        if abs(dr) >= abs(dc):
            return 0 if dr > 0 else 1   # down or up
        else:
            return 2 if dc > 0 else 3   # right or left

    # combine seed and magic maps for nearest goal
    goal_map = np.clip(seed_map + magic_map, 0, 1)
    goal_dir = nearest_dir(goal_map, row, col)
    trap_dir = nearest_dir(trap_map, row, col)

    return (row, col, goal_dir, trap_dir)

def get_q_values(q_table, state):
    if state not in q_table:
        q_table[state] = np.zeros(4) 
    return q_table[state]

def train(shaped_reward=False, seed=42):
    np.random.seed(seed)
    env = HamsterEnv(grid_size=5, shaped_reward=shaped_reward, max_steps=MAX_STEPS)

    q_table = {}  
    eps     = EPS_START

    all_rewards = []
    all_steps   = []
    all_wins    = []

    for ep in range(EPISODES):
        obs, info = env.reset()
        state     = get_state(obs)
        total_r   = 0.0
        won       = False

        for _ in range(MAX_STEPS):
            if np.random.random() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(get_q_values(q_table, state)))

            next_obs, reward, done, truncated, info = env.step(action)
            next_state = get_state(next_obs)

            q_now  = get_q_values(q_table, state)
            q_next = get_q_values(q_table, next_state)
            q_now[action] += LR * (reward + GAMMA * np.max(q_next) - q_now[action])

            state    = next_state
            total_r += reward

            if done or truncated:
                won = info.get("win", False)
                break

        eps = max(EPS_END, eps * EPS_DECAY)

        all_rewards.append(total_r)
        all_steps.append(info["steps"])
        all_wins.append(int(won))

        if (ep + 1) % 5000 == 0:
            recent_reward   = np.mean(all_rewards[-500:])
            recent_win_rate = np.mean(all_wins[-500:]) * 100
            print(f"  ep {ep+1:>6} | avg reward: {recent_reward:>7.2f} | win rate: {recent_win_rate:.1f}% | eps: {eps:.3f}")

    env.close()

    label = "shaped" if shaped_reward else "sparse"
    with open(f"q_table_{label}.pkl", "wb") as f:
        pickle.dump(q_table, f)

    logs = {"reward": all_rewards, "steps": all_steps, "wins": all_wins}
    np.save(f"ql_logs_{label}.npy", logs)
    print(f"\nSaved q_table_{label}.pkl and ql_logs_{label}.npy")

    return q_table, logs


# evaluation
def evaluate(q_table, shaped_reward=False, n_episodes=500):
    import time
    env     = HamsterEnv(grid_size=5, shaped_reward=shaped_reward)
    rewards = []
    wins    = []
    steps   = []
    times   = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        state = get_state(obs)
        total_r = 0.0

        for _ in range(MAX_STEPS):
            t0 = time.perf_counter()
            action = int(np.argmax(get_q_values(q_table, state)))
            times.append(time.perf_counter() - t0)

            obs, r, done, truncated, info = env.step(action)
            state = get_state(obs)
            total_r += r
            if done or truncated:
                break

        rewards.append(total_r)
        wins.append(int(info.get("win", False)))
        steps.append(info["steps"])

    env.close()

    results = {
        "avg_reward":round(float(np.mean(rewards)), 3),
        "win_rate_%": round(float(np.mean(wins)) * 100, 2),
        "avg_steps":round(float(np.mean(steps)), 2),
        "avg_inference_ms": round(float(np.mean(times)) * 1000, 4),
    }

    print("\n Q-Learning Eval ")
    for k, v in results.items():
        print(f"  {k:<22}: {v}")

    return results


# main
if __name__ == "__main__":
    print("Training Q-Learning (sparse reward) ...")
    q_table, logs = train(shaped_reward=False)
    evaluate(q_table)

    print("\nTraining Q-Learning (shaped reward) ...")
    q_table_shaped, logs_shaped = train(shaped_reward=True)
    evaluate(q_table_shaped, shaped_reward=True)