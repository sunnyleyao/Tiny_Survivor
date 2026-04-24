import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import torch
import os

from hamster_env import HamsterEnv, EMPTY, SEED, MAGIC, TAPE, STACK
from q_learning import get_state, get_q_values, MAX_STEPS
from dqn import QNetwork, device, load_model

def run_episode(agent_type, model_or_table, shaped=False, seed=None):
    env = HamsterEnv(grid_size=5, shaped_reward=shaped)
    obs, info = env.reset(seed=seed)
    trajectory = []

    for _ in range(MAX_STEPS):
        state_snapshot = env.get_state() 

        if agent_type == "ql":
            s     = get_state(obs)
            action = int(np.argmax(get_q_values(model_or_table, s)))
        else:
            s_t    = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = int(model_or_table(s_t).argmax().item())

        obs, reward, done, truncated, info = env.step(action)
        trajectory.append({
            "snapshot": state_snapshot,
            "action":   action,
            "reward":   reward,
            "info":     info.copy(),
        })

        if done or truncated:
            break

    env.close()
    return trajectory, info

def behavioral_analysis(agent_type, model_or_table):
    print(f"\n Behavioral Analysis ({agent_type.upper()})")

    ACTION_NAMES = ["up", "down", "left", "right"]

    # scenario 1: trap right next to a seed
    # Seed at (0,1), Trap at (0,2), hamster at (0,0)")
    # does the agent go for the seed even with a trap nearby?

    env = HamsterEnv(grid_size=5)
    env.reset(seed=0)

    env.grid[:] = EMPTY
    env.grid[0, 1] = SEED
    env.grid[0, 2] = TAPE
    env.hamster_pos = (0, 0)
    env.score       = 20
    env.items_left  = 1

    obs = env._get_obs()
    action = _get_action(agent_type, model_or_table, obs)
    print(f"  Agent chose: {ACTION_NAMES[action]}", end="  -->  ")
    if action == 3:   # right = toward seed
        print("Goes for the seed")
    elif action == 1:
        print("Goes down (avoiding the row entirely)")
    else:
        print(f"Chose {ACTION_NAMES[action]} (cautious or confused)")
    env.close()

    # scenario 2: one item left, far corner
    # Only Helsius left at (4,4), hamster at (0,0)
    # does it head toward the goal?

    env = HamsterEnv(grid_size=5)
    env.reset(seed=0)
    env.grid[:] = EMPTY
    env.grid[4, 4] = MAGIC
    env.hamster_pos = (0, 0)
    env.score = 20
    env.items_left = 1

    obs    = env._get_obs()
    action = _get_action(agent_type, model_or_table, obs)
    print(f"Agent chose: {ACTION_NAMES[action]}", end="  -->  ")
    if action in (1, 3):   # down or right = toward (4,4)
        print("Moving toward the goal")
    else:
        print(f"Chose {ACTION_NAMES[action]} (moving away from goal)")
    env.close()

    # scenario 3: very low score, trap nearby
    # Score=3, trap at (0,1), seed at (1,0), hamster at (0,0)")
    # does low score make it more conservative?

    env = HamsterEnv(grid_size=5)
    env.reset(seed=0)
    env.grid[:] = EMPTY
    env.grid[0, 1] = TAPE
    env.grid[1, 0] = SEED
    env.hamster_pos = (0, 0)
    env.score = 3    # one more trap = game over
    env.items_left = 1

    obs    = env._get_obs()
    action = _get_action(agent_type, model_or_table, obs)
    print(f"  Agent chose: {ACTION_NAMES[action]}", end="  -->  ")
    if action == 1:   # down = toward seed, away from trap
        print("Goes for seed safely (avoids trap)")
    elif action == 3:
        print("Walks into the trap")
    else:
        print(f"? Chose {ACTION_NAMES[action]}")
    env.close()



def _get_action(agent_type, model_or_table, obs):
    if agent_type == "ql":
        s = get_state(obs)
        return int(np.argmax(get_q_values(model_or_table, s)))
    else:
        s_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            return int(model_or_table(s_t).argmax().item())



def simulation_eval(agent_type, model_or_table, n_episodes=300):
    
    print(f"\n Simulation Eval ({agent_type.upper()}, {n_episodes} episodes)")

    all_rewards = []
    all_items = []
    all_traps_hit = []
    all_steps = []
    all_wins = []

    for ep in range(n_episodes):
        env = HamsterEnv(grid_size=5)
        obs, info = env.reset(seed=ep)

        total_r    = 0.0
        items_got  = 0
        traps_hit  = 0

        for _ in range(MAX_STEPS):
            action = _get_action(agent_type, model_or_table, obs)
            obs, reward, done, truncated, info = env.step(action)
            total_r += reward

            if reward >= 5:    # picked up seed or magic
                items_got += 1
            if reward <= -5:   # hit a trap 
                traps_hit += 1

            if done or truncated:
                break

        env.close()
        all_rewards.append(total_r)
        all_items.append(items_got)
        all_traps_hit.append(traps_hit)
        all_steps.append(info["steps"])
        all_wins.append(int(info.get("win", False)))

    print(f" avg reward: {np.mean(all_rewards):.2f}")
    print(f"  win rate:        {np.mean(all_wins)*100:.1f}%")
    print(f"  avg items got:   {np.mean(all_items):.2f} / 4")
    print(f"  avg traps hit:   {np.mean(all_traps_hit):.2f}")
    print(f"  avg steps:       {np.mean(all_steps):.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"{agent_type.upper()} -- Simulation Results ({n_episodes} eps)")

    axes[0].hist(all_rewards, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_xlabel("Total Reward")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Reward Distribution")

    axes[1].hist(all_items, bins=[0,1,2,3,4,5], align="left",
                 color="mediumseagreen", edgecolor="white", rwidth=0.7)
    axes[1].set_xlabel("Items Collected")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Items Collected per Episode")
    axes[1].set_xticks([0,1,2,3,4])

    plt.tight_layout()
    fname = f"simulation_{agent_type}.png"
    plt.savefig(fname, dpi=150)
    plt.close()

    return {
        "avg_reward": round(np.mean(all_rewards), 2),
        "win_rate": round(np.mean(all_wins) * 100, 1),
        "avg_items": round(np.mean(all_items), 2),
        "avg_traps_hit": round(np.mean(all_traps_hit), 2),
        "avg_steps": round(np.mean(all_steps), 1),
    }



