## AI Tool Usage

## `hamster_env.py`

**What AI did**
I asked AI to provide a basic Gymnasium environment scaffold, specifically, telling me which methods are required by the Gymnasium API 
and what each one needs to return. 
AI also wrote the render() and get_state() functions, and designed the _get_obs() observation vector structure.

**What I did**
All visual assets were hand-drawn by me

I implemented the game-specific logic, like the item placement (_place_items()), the movement and wall blocking in step(), the item interaction and score tracking, the win/lose conditions, and the reward values for each item type. 
I used AI-generated scaffold as a starting point and filled in that makes this a hamster game rather than a generic environment.

And in the beginning I initiated the starting score as 20. But he initial score of 20 caused agents to die within 5-10 steps before collecting
any items, so the reward signal was too sparse to learn. I changed it to 50 after observing an almost zero win rates in early training runs.

---

## `q_learning.py`

**What AI did**
AI suggested two helper functions I needed but hadn't thought of: nearest_dir() (to compute which direction the nearest goal or trap is from the hamster's current position) and get_q_values() (to look up Q-values from the table, defaulting to zeros for unseen states). 
AI also helped me debug my training function, clean up my style, and add the training progress logging and model saving block at the end of the training loop.
AI also provided the orginal version of `get_state()`function

**What I did**
I wrote the q-learning training function loop and implemented the two helper function with AI's suggestions.
I rewrote the `get_state()` function provided by Ai.
In the original version, it sed full binary maps for seed, magic, and trap locations (75 binary values). 
State space was 2^75. Thus, the agent almost never visited the same state twice. So Q-values never updated meaningfully. Win rate was stuck at 1%.
Then I simplified to just (row, col, seed_map, magic_map). But it was still too large, Q-table grew to 500K+ entries.
I redesigned it to encode only (row, col, goal_direction, trap_direction) with just 4 values giving about 625 total states. 
Then the win rate jumped to 94%.

I also increased `LR` from 0.1 to 0.5 and slowed `EPS_DECAY` from 0.9995 to 0.9998
after observing that the agent stopped exploring too early.

---

## `dqn.py`

**What AI did**
`ReplayBuffer` class, and the overall DQN training loop skeleton including the target network synchronization pattern.

**What I did**
I wrote the `QNetwork` class structure and implemented the training based on the skeleton provided by AI.
Specifically, I  fixed the scheduler order bug as the initial version provided by AI called `scheduler.step()` before
`optimizer.step()`, causing a PyTorch warning and skipping the first LR value. I fixed the order and added a condition 
to only step the scheduler when there were actual losses in that episode.

And during the training I found DQN peaked at 77% win rate at episode 5000 then
dropped back to 3% as training continued. I found the epsilon had decayed too fast, causing the network to overfit.
So I changed `EPS_DECAY` from 0.9995 to 0.9998 and reduced network size from 128 to 64 hidden units. 
I also just saved the checkpoint at episode 5,000 rather than continuing training.
I also Changed `LR` from 1e-3 to 5e-4, `BATCH_SIZE` from 64 to 32, `BUFFER_SIZE` from 10000 to 5000, and `TARGET_UPDATE` from 200 to 100
after observing unstable training in early runs.

---

## `compare.py`

**What AI did**
The smooth() helper function, and the code for generating the learning curve plots, win rate plots, and the 2×2 ablation table printout.

**What I did**
I made the overall structure of the comparison. Specifically, I designed the exact experiments to run (4 combinations: 2 algorithms × 2 reward types) 
And I designed The 2×2 ablation table format
I also decided which metrics to report (avg reward, win rate, avg steps) 
AI provided the code to execute my design, but the research question and experiment structure came from me.

---

## `evaluate.py`

**What AI did**
`main()`, `run_episode()` helper and the `behavioral_analysis()` function. 
Finish the plotting code for the reward distribution and items collected histograms, and the results dictionary formatting 
at the end in simulation.

**What I did**
I designed the 3 specific behavioral analysis scenarios in `behavioral_analysis()`
The first one is making trap next to a seed to test risk tolerance.
The second one is to put a single item in far corner to test navigation
And the third one is to us very low score with nearby trap to test conservative behavior under pressure
I wrote main simulation loop structure with running multiple episodes, tracking metrics per episode, and the logic for categorizing results.

---

## `app.py`

**What AI did**
The Streamlit app skeleton and the `draw_grid()` HTML table renderer.

**What I did**
I made visual design decisions and adjust these myself based on the skeleton provided by AI.
I thought of the game title "Tiny Survivor" and Toto's character description
I integrated my hand-drawn assets via base64 encoding, replacing the emoji version provided by AI.

Initially, I tried to use Streamlit Cloud to make the website public. But Streamlit Cloud used Python 3.14 which doesn't support torch.
So I spent significant time debugging this before deciding to keep the app local-only.

## `analysis.ipynb`

**What AI did**
Generated the code of plotting.

**What I did**
I decided what each sections should be included in the analysis and what kind of plot should be used.

## `README.md`
AI helped me summerize the quantitative results from several results I got from different models and evaluations.



---

## External Libraries Used

| Library | Use |
|---|---|
| gymnasium | RL environment API | 
| torch | DQN neural network | 
| numpy | Array operations | 
| matplotlib | Plotting learning curves | 
| streamlit | Web app framework | 
| pickle | Saving Q-table |

---

