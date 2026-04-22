"""
app.py

This is the Streamlit web app for my Hamster RL project.
The user can pick which trained agent to watch (Q-Learning or DQN),
then see it play the game live on the map.

This covers:
  - Deployed model as functional web application with UI (10 pts)
  - Real-time ML inference with interactive application (7 pts)

To run:
  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pickle
import torch
import time
import os

from hamster_env import HamsterEnv, EMPTY, SEED, MAGIC, TAPE, STACK
from q_learning import get_state, get_q_values, MAX_STEPS
from dqn import QNetwork, load_model, device

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🐹 Hamster RL",
    page_icon="🐹",
    layout="centered",
)

st.title("🐹 Hamster RL")
st.markdown("""
Watch a trained RL agent control a hamster on a 5×5 grid.
The hamster collects **seeds (+5)** and **magic (+10)** while avoiding
**tape (-5)** and **stacks (-5)**. Each step also costs **-1**.

Win condition: collect all seeds and magic before score hits 0.
""")

# ── sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Settings")

agent_choice = st.sidebar.radio(
    "Choose agent:",
    ["Q-Learning", "DQN"],
)

reward_choice = st.sidebar.radio(
    "Reward type:",
    ["Sparse", "Shaped"],
)

speed = st.sidebar.slider(
    "Playback speed (steps/sec)",
    min_value=1, max_value=10, value=4
)

episode_seed = st.sidebar.number_input(
    "Episode seed (controls map layout)",
    min_value=0, max_value=9999, value=42
)

run_button = st.sidebar.button("▶ Run Episode")


# ── load models (cached so we don't reload every frame) ──────────────────────
@st.cache_resource
def load_ql(variant):
    path = f"q_table_{variant}.pkl"
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_dqn(variant):
    path = f"dqn_{variant}.pth"
    if not os.path.exists(path):
        return None
    obs_dim = HamsterEnv().observation_space.shape[0]
    return load_model(path, obs_dim)


# ── draw the grid as an HTML table ───────────────────────────────────────────
CELL_EMOJI = {
    EMPTY: "⬜",
    SEED:  "assets/seed.png",
    MAGIC: "✨",
    TAPE:  "📼",
    STACK: "assets/stack.png",
}
HAMSTER_IMG = "assets/hamster.png"

def draw_grid(grid, hamster_pos,use_images=True):
    """
    Render the 5x5 grid as a simple HTML table with emojis.
    Streamlit can display raw HTML so this gives a clean visual
    without needing pygame.
    """
    html = """
    <style>
      table.hamster { border-collapse: collapse; margin: auto; }
      table.hamster td {
        width: 60px; height: 60px;
        text-align: center; font-size: 28px;
        border: 2px solid #ccc;
        background: #fdf6e3;
        border-radius: 6px;
      }
      table.hamster td.hamster-cell { background: #ffe0b2; }
    </style>
    <table class="hamster">
    """
    for r in range(grid.shape[0]):
        html += "<tr>"
        for c in range(grid.shape[1]):
            is_hamster = (r, c) == hamster_pos
            cell_class = "hamster-cell" if is_hamster else ""
            if is_hamster:
                emoji = "🐹"
            else:
                emoji = CELL_EMOJI.get(grid[r, c], "⬜")
            html += f'<td class="{cell_class}">{emoji}</td>'
        html += "</tr>"
    html += "</table>"
    return html


# ── main episode runner ───────────────────────────────────────────────────────
def run_episode(agent_type, model_or_table, shaped, seed, delay):
    """
    Run one episode step by step, updating the Streamlit UI each step.
    """
    env = HamsterEnv(grid_size=5, shaped_reward=shaped)
    obs, info = env.reset(seed=int(seed))

    # placeholders so we can update them in place each step
    grid_placeholder  = st.empty()
    stats_placeholder = st.empty()
    log_placeholder   = st.empty()

    ACTION_NAMES = ["⬆ up", "⬇ down", "⬅ left", "➡ right"]
    step_log     = []
    total_r      = 0.0

    for step in range(MAX_STEPS):

        # get action from agent
        if agent_type == "ql":
            s      = get_state(obs)
            action = int(np.argmax(get_q_values(model_or_table, s)))
        else:
            s_t    = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = int(model_or_table(s_t).argmax().item())

        # draw current state BEFORE taking the step
        state  = env.get_state()
        html   = draw_grid(state["grid"], state["hamster_pos"])
        grid_placeholder.markdown(html, unsafe_allow_html=True)

        stats_placeholder.markdown(f"""
        | Step | Score | Items Left |
        |------|-------|------------|
        | {state['steps']} | {state['score']} | {state['items_left']} |
        """)

        # take the step
        obs, reward, done, truncated, info = env.step(action)
        total_r += reward

        step_log.append(f"Step {step+1}: {ACTION_NAMES[action]}  |  reward: {reward:+.0f}  |  score: {info['score']}")

        # show last 5 steps in the log
        log_placeholder.code("\n".join(step_log[-5:]))

        time.sleep(1.0 / delay)

        if done or truncated:
            break

    # draw final state
    final_state = env.get_state()
    grid_placeholder.markdown(
        draw_grid(final_state["grid"], final_state["hamster_pos"]),
        unsafe_allow_html=True
    )
    env.close()

    # result banner
    if info.get("win"):
        st.success(f"🎉 Win! Final score: {info['score']}  |  Total reward: {total_r:.1f}")
    elif info.get("lose"):
        st.error(f"💀 Lost (score hit 0). Total reward: {total_r:.1f}")
    else:
        st.warning(f"⏱ Time limit reached. Items left: {info['items_left']}. Total reward: {total_r:.1f}")

    return total_r, info


# ── main UI logic ─────────────────────────────────────────────────────────────
variant = reward_choice.lower()   # "sparse" or "shaped"

if run_button:
    # load the selected agent
    if agent_choice == "Q-Learning":
        model = load_ql(variant)
        atype = "ql"
    else:
        model = load_dqn(variant)
        atype = "dqn"

    if model is None:
        st.error(f"Model file not found. Please train the {agent_choice} agent first by running:")
        if agent_choice == "Q-Learning":
            st.code("python q_learning.py")
        else:
            st.code("python dqn.py")
    else:
        st.subheader(f"Running: {agent_choice} ({reward_choice} reward)  |  Seed: {episode_seed}")
        shaped = (variant == "shaped")
        run_episode(atype, model, shaped, episode_seed, speed)

else:
    # show a placeholder grid before the user presses run
    st.markdown("### Press ▶ Run Episode to start")

    env = HamsterEnv(grid_size=5)
    obs, _ = env.reset(seed=int(episode_seed))
    state  = env.get_state()
    st.markdown(
        draw_grid(state["grid"], state["hamster_pos"]),
        unsafe_allow_html=True
    )
    env.close()

    st.markdown("""
    ---
    **Legend:**
    🐹 Hamster &nbsp;&nbsp;
    🌱 Seed (+5) &nbsp;&nbsp;
    ✨ Magic (+10) &nbsp;&nbsp;
    📼 Tape (-5) &nbsp;&nbsp;
    🪨 Stack (-5) &nbsp;&nbsp;
    ⬜ Empty
    """)