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
import base64

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.hamster_env import HamsterEnv, EMPTY, SEED, MAGIC, TAPE, STACK
from src.q_learning import get_state, get_q_values, MAX_STEPS
from src.dqn import QNetwork, load_model, device

# ── page config 
st.set_page_config(
    page_title="🐹 Tiny Survivor",
    page_icon="🐹",
    layout="centered",
)

# ── global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* warm parchment background */
  .stApp { background-color: #fdf6e3; }

  /* sidebar */
  [data-testid="stSidebar"] { background-color: #fef9f0; border-right: 2px solid #e8d5b7; }

  /* title */
  h1 { color: #7b4f2e !important; letter-spacing: 1px; }

  /* section headers */
  h3 { color: #9b6b3a !important; }

  /* metric cards */
  [data-testid="stMetricValue"] { font-size: 2rem !important; color: #7b4f2e !important; }
  [data-testid="stMetricLabel"] { color: #a0785a !important; font-weight: 600 !important; }

  /* legend card */
  .legend-card {
    background: #fff8ec;
    border: 2px solid #e8d5b7;
    border-radius: 12px;
    padding: 12px 20px;
    margin-top: 10px;
    line-height: 2;
  }

  /* run button */
  [data-testid="stSidebar"] .stButton > button {
    background-color: #c77b3b;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    width: 100%;
    padding: 10px;
    transition: background 0.2s;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
    background-color: #a0602c;
    color: white;
  }

  /* divider */
  hr { border-color: #e8d5b7; }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.title("🐹 Tiny Survivor")
st.markdown("""
*Help me! I'm **Toto**, and I'm stuck in this box.*
Watch me explore, collect seeds and Helsius Energy Drink, and dodge the cat's paw and the bin. **Cheer for me!**

**Win condition:** collect all seeds & magic before score hits 0.
""")
st.divider()

# sidebar controls 
with st.sidebar:
    st.markdown("## Settings")
    st.divider()

    agent_choice = st.radio(
        "Agent",
        ["Q-Learning", "DQN"],
        help="Q-Learning uses a lookup table; DQN uses a neural network.",
    )

    reward_choice = st.radio(
        "Reward type",
        ["Sparse", "Shaped"],
        help="Sparse: only win/lose rewards. Shaped: step-by-step hints.",
    )

    st.divider()

    speed = st.slider(
        "Playback speed (steps/sec)",
        min_value=1, max_value=10, value=4,
    )

    episode_seed = st.number_input(
        "Episode seed",
        min_value=0, max_value=9999, value=42,
        help="Controls the random map layout.",
    )

    st.divider()
    run_button = st.button("▶ Run Episode")


#  load models (cached so we don't reload every frame)
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


# grid renderer
def load_img_base64(path):
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = path.split(".")[-1]
    return f"data:image/{ext};base64,{data}"

def draw_grid(grid, hamster_pos):
    CELL_IMG = {
        EMPTY: load_img_base64("assets/empty.png"),
        SEED:  load_img_base64("assets/seed.png"),
        MAGIC: load_img_base64("assets/magic.png"),
        TAPE:  load_img_base64("assets/tape.png"),
        STACK: load_img_base64("assets/stack.png"),
    }
    HAMSTER_B64 = load_img_base64("assets/hamster.png")

    def cell_content(cell_type, is_hamster):
        if is_hamster:
            return f'<img src="{HAMSTER_B64}" width="90" height="90" style="object-fit:contain;">'
        src = CELL_IMG.get(cell_type, CELL_IMG[EMPTY])
        return f'<img src="{src}" width="90" height="90" style="object-fit:contain;">'

    html = """
    <style>
      table.hamster {
        border-collapse: separate;
        border-spacing: 6px;
        margin: auto;
      }
      table.hamster td {
        width: 100px; height: 100px;
        text-align: center; font-size: 32px;
        border: 2px solid #d4b896;
        background: #fffdf7;
        border-radius: 12px;
        padding: 4px;
        box-shadow: 2px 2px 4px rgba(0,0,0,0.08);
        transition: background 0.2s;
      }
      table.hamster td.hamster-cell {
        background: #ffe0b2;
        border-color: #c77b3b;
        box-shadow: 0 0 8px rgba(199,123,59,0.35);
      }
    </style>
    <table class="hamster">
    """
    for r in range(grid.shape[0]):
        html += "<tr>"
        for c in range(grid.shape[1]):
            is_hamster = (r, c) == hamster_pos
            cell_class = "hamster-cell" if is_hamster else ""
            content    = cell_content(grid[r, c], is_hamster)
            html += f'<td class="{cell_class}">{content}</td>'
        html += "</tr>"
    html += "</table>"
    return html


# ── legend ────────────────────────────────────────────────────────────────────
def show_legend():
    st.markdown("""
<div class="legend-card">
<b>Legend</b><br>
🐹 <b>Toto</b> &nbsp;|&nbsp;
🌱 <b>Seed</b> +5 &nbsp;|&nbsp;
✨ <b>Magic</b> +10 &nbsp;|&nbsp;
📼 <b>Tape</b> −5 &nbsp;|&nbsp;
🪨 <b>Stack</b> −5 &nbsp;|&nbsp;
⬜ Empty
</div>
""", unsafe_allow_html=True)


# ── episode runner
def run_episode(agent_type, model_or_table, shaped, seed, delay):
    env = HamsterEnv(grid_size=5, shaped_reward=shaped)
    obs, info = env.reset(seed=int(seed))

    grid_placeholder  = st.empty()

    col1, col2, col3 = st.columns(3)
    step_metric  = col1.empty()
    score_metric = col2.empty()
    items_metric = col3.empty()

    log_placeholder = st.empty()

    ACTION_NAMES = ["⬆ Up", "⬇ Down", "⬅ Left", "➡ Right"]
    step_log = []
    total_r  = 0.0

    for step in range(MAX_STEPS):
        if agent_type == "ql":
            s      = get_state(obs)
            action = int(np.argmax(get_q_values(model_or_table, s)))
        else:
            s_t    = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                action = int(model_or_table(s_t).argmax().item())

        state = env.get_state()
        grid_placeholder.markdown(
            draw_grid(state["grid"], state["hamster_pos"]),
            unsafe_allow_html=True,
        )

        step_metric.metric("Step",       state["steps"])
        score_metric.metric("Score",     state["score"])
        items_metric.metric("Items Left", state["items_left"])

        obs, reward, done, truncated, info = env.step(action)
        total_r += reward

        step_log.append(
            f"Step {step+1:>3}  {ACTION_NAMES[action]:<10}  "
            f"reward: {reward:+.0f}   score: {info['score']}"
        )
        log_placeholder.code("\n".join(step_log[-6:]), language=None)

        time.sleep(1.0 / delay)
        if done or truncated:
            break

    # final frame
    final_state = env.get_state()
    grid_placeholder.markdown(
        draw_grid(final_state["grid"], final_state["hamster_pos"]),
        unsafe_allow_html=True,
    )
    env.close()

    st.divider()
    if info.get("win"):
        st.success(f"🎉 **Toto wins!**  Final score: {info['score']}  |  Total reward: {total_r:.1f}")
    elif info.get("lose"):
        st.error(f"💀 **Toto lost** (score hit 0).  Total reward: {total_r:.1f}")
    else:
        st.warning(f"⏱ **Time limit reached.**  Items left: {info['items_left']}  |  Total reward: {total_r:.1f}")

    return total_r, info


# ── main UI
variant = reward_choice.lower()

if run_button:
    if agent_choice == "Q-Learning":
        model = load_ql(variant)
        atype = "ql"
    else:
        model = load_dqn(variant)
        atype = "dqn"

    if model is None:
        st.error(f"Model file not found for **{agent_choice}** ({reward_choice}). Train it first:")
        if agent_choice == "Q-Learning":
            st.code("python q_learning.py")
        else:
            st.code("python dqn.py")
    else:
        st.markdown(f"### {agent_choice} · {reward_choice} reward · Seed {episode_seed}")
        shaped = (variant == "shaped")
        run_episode(atype, model, shaped, episode_seed, speed)

else:
    st.markdown("### Press **▶ Run Episode** in the sidebar to start!")

    env = HamsterEnv(grid_size=5)
    obs, _ = env.reset(seed=int(episode_seed))
    state  = env.get_state()
    st.markdown(
        draw_grid(state["grid"], state["hamster_pos"]),
        unsafe_allow_html=True,
    )
    env.close()

    show_legend()
