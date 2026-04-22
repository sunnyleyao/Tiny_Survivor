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

#page
st.set_page_config(
    page_title="🐹 Tiny Survivor",
    page_icon="🐹",
    layout="centered",
)

st.title("🐹 Tiny Survivor")
st.markdown("""
Help me! I’m Toto, and I’m stuck in this box.
Watch me explore, collect seeds and Helsius Energy Drink,
and dodge the cat’s paw and the bin.
Cheer for me!

Win condition: collect all seeds and magic before score hits 0.
""")

# sidebar controls
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


# draw the grid
import base64

def load_img_base64(path):
    """Convert local image to base64 so it can be embedded in HTML."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = path.split(".")[-1]
    return f"data:image/{ext};base64,{data}"

def draw_grid(grid, hamster_pos, use_images=False):
    """
    Render the 5x5 grid as an HTML table.

    use_images=False  -> uses emojis (default, no setup needed)
    use_images=True   -> uses your own images from assets/ folder
                         just drop your png files in assets/ and set this to True
    """
    if use_images:
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
                return f'<img src="{HAMSTER_B64}" width="100" height="100" style="object-fit:contain;">'
            src = CELL_IMG.get(cell_type, CELL_IMG[EMPTY])
            return f'<img src="{src}" width="100" height="100" style="object-fit:contain;">'
    else:
        CELL_EMOJI = {
            EMPTY: "⬜",
            SEED:  "🌱",
            MAGIC: "✨",
            TAPE:  "📼",
            STACK: "🪨",
        }
        def cell_content(cell_type, is_hamster):
            if is_hamster:
                return "🐹"
            return CELL_EMOJI.get(cell_type, "⬜")

    html = """
    <style>
      table.hamster { border-collapse: collapse; margin: auto; }
      table.hamster td {
        width: 110px; height: 110px;
        text-align: center; font-size: 28px;
        border: 3px solid #d4b896;
        background: #fdf6e3;
        border-radius: 8px;
        padding: 5px;
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
            content    = cell_content(grid[r, c], is_hamster)
            html += f'<td class="{cell_class}">{content}</td>'
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
        html   = draw_grid(state["grid"], state["hamster_pos"], use_images=True)
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
        draw_grid(final_state["grid"], final_state["hamster_pos"], use_images=True),
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
        draw_grid(state["grid"], state["hamster_pos"], use_images=True),
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