import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
TORCH_AVAILABLE = True

from collections import deque
import random
from hamster_env import HamsterEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if TORCH_AVAILABLE else None

EPISODES      = 50000
LR            = 5e-4    # lower LR = more stable training
GAMMA         = 0.95
EPS_START     = 1.0
EPS_END       = 0.05
EPS_DECAY     = 0.9998  # much slower decay -- agent explores for longer
BATCH_SIZE    = 32      # smaller batch = more frequent updates
BUFFER_SIZE   = 5000    # smaller buffer = more recent experience
TARGET_UPDATE = 100     # update target network more frequently
MAX_STEPS     = 200


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.network(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch                              = random.sample(self.memory, batch_size)
        states, actions, rewards, nexts, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(nexts)).to(device),
            torch.FloatTensor(dones).to(device),
        )

    def __len__(self):
        return len(self.memory)


# training 
def train(shaped_reward=False, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env     = HamsterEnv(grid_size=5, shaped_reward=shaped_reward, max_steps=MAX_STEPS)
    obs_dim = env.observation_space.shape[0]

    # main network: the one we actually train
    main_net   = QNetwork(obs_dim).to(device)

    # target network: a frozen copy we use to compute TD targets
    # we only update this every TARGET_UPDATE steps to keep targets stable
    target_net = QNetwork(obs_dim).to(device)
    target_net.load_state_dict(main_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(main_net.parameters(), lr=LR)

    # cosine annealing scheduler -- gradually reduces LR over training (3 pts)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPISODES)

    buffer     = ReplayBuffer(BUFFER_SIZE)
    eps        = EPS_START
    step_count = 0

    all_rewards = []
    all_steps   = []
    all_wins    = []
    all_losses  = []

    for ep in range(EPISODES):
        obs, info = env.reset()
        state     = obs.copy()
        total_r   = 0.0
        won       = False
        ep_losses = []

        for _ in range(MAX_STEPS):

            # epsilon-greedy action selection
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    s_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action   = int(main_net(s_tensor).argmax().item())

            next_obs, reward, done, truncated, info = env.step(action)

            # store this transition in the replay buffer
            buffer.store(state, action, reward, next_obs, float(done))
            state      = next_obs.copy()
            total_r   += reward
            step_count += 1

            # only start training once we have enough samples in the buffer
            if len(buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

                # what the main network currently predicts for Q(s,a)
                q_predicted = main_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # what the target network says the best next Q-value is
                with torch.no_grad():
                    q_next_max = target_net(next_states).max(1)[0]
                    q_target   = rewards + GAMMA * q_next_max * (1 - dones)

                loss = nn.MSELoss()(q_predicted, q_target)

                optimizer.zero_grad()
                loss.backward()

                # gradient clipping: stops gradients from getting too large (3 pts)
                nn.utils.clip_grad_norm_(main_net.parameters(), max_norm=1.0)

                optimizer.step()
                ep_losses.append(loss.item())

            # copy main network weights to target network periodically
            if step_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(main_net.state_dict())

            if done or truncated:
                won = info.get("win", False)
                break

        # decay epsilon and step the LR scheduler
        # note: scheduler.step() must come after optimizer.step()
        eps = max(EPS_END, eps * EPS_DECAY)
        if len(ep_losses) > 0:
            scheduler.step()

        all_rewards.append(total_r)
        all_steps.append(info["steps"])
        all_wins.append(int(won))
        all_losses.append(float(np.mean(ep_losses)) if ep_losses else 0.0)

        if (ep + 1) % 5000 == 0:
            recent_reward   = np.mean(all_rewards[-500:])
            recent_win_rate = np.mean(all_wins[-500:]) * 100
            print(f"  ep {ep+1:>6} | avg reward: {recent_reward:>7.2f} | win rate: {recent_win_rate:.1f}% | eps: {eps:.3f}")

    env.close()

    label = "shaped" if shaped_reward else "sparse"
    torch.save(main_net.state_dict(), f"dqn_{label}.pth")

    logs = {
        "reward": all_rewards,
        "steps":  all_steps,
        "wins":   all_wins,
        "loss":   all_losses,
    }
    np.save(f"dqn_logs_{label}.npy", logs)
    print(f"\nSaved dqn_{label}.pth and dqn_logs_{label}.npy")

    return main_net, logs


# evaluation 
def evaluate(model, shaped_reward=False, n_episodes=500):
    import time
    env     = HamsterEnv(grid_size=5, shaped_reward=shaped_reward)
    model.eval()

    rewards = []
    wins    = []
    steps   = []
    times   = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        total_r   = 0.0

        for _ in range(MAX_STEPS):
            s_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                action = int(model(s_tensor).argmax().item())
            times.append(time.perf_counter() - t0)

            obs, r, done, truncated, info = env.step(action)
            total_r += r
            if done or truncated:
                break

        rewards.append(total_r)
        wins.append(int(info.get("win", False)))
        steps.append(info["steps"])

    env.close()

    results = {
        "avg_reward":       round(float(np.mean(rewards)), 3),
        "win_rate_%":       round(float(np.mean(wins)) * 100, 2),
        "avg_steps":        round(float(np.mean(steps)), 2),
        "avg_inference_ms": round(float(np.mean(times)) * 1000, 4),
    }

    print("\n DQN Eval")
    for k, v in results.items():
        print(f"  {k:<22}: {v}")

    return results


def load_model(path, obs_dim):
    """Helper to reload a saved model."""
    if not TORCH_AVAILABLE:
        return None
    model = QNetwork(obs_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# main
if __name__ == "__main__":
    print("Training DQN (sparse reward)")
    model, logs = train(shaped_reward=False)
    evaluate(model)

    print("\nTraining DQN (shaped reward)")
    model_shaped, logs_shaped = train(shaped_reward=True)
    evaluate(model_shaped, shaped_reward=True)