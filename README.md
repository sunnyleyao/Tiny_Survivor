# Tiny Survivor
This is a reinforcement learning game. The game happens in a custom grid-world where a hamster learns to collect food and avoid traps using two RL algorithms: tabular Q-Learning and Deep Q-Network (DQN). No rules were given to the agents, and they learned entirely through trial and error.

## What it Does
Tiny Survivor is a 5×5 grid-world game built from a custom Gymnasium environment. A hamster named Toto starts in the top-left corner and must collect all seeds (+5) and a Helsius energy drink (+10) while avoiding a cat's paw (-5) and a bin (-5). The score starts at 50 which tracks both the hamster's health and its progress through the game. Every step costs 1 point, and the game ends when Toto collects everything or the score hits zero. Two RL agents were trained to play the game. The first is a tabular Q-Learning agent that learns a lookup table of optimal actions, and the other is a DQN agent that uses a neural network to approximate Q-values. And agents are trained under two reward designs which are sparse reward (only item pickups and traps) and shaped reward (with an additional bonus for moving toward the nearest goal). The project mainly compares these two approaches and investigates whether extra guidance helps either agent learn faster. It compares all four combinations and draws conclusions about which algorithm and reward design works best in this environment.

**Research Problem**: *In the same custom environment, does DQN actually learn better than tabular Q-learning, and does reward shaping help either of them?*

## Quick Start
Type the following in the terminal, and you will directly open the website application of this game. 
```bash
git clone https://github.com/sunnyleyao/Tiny_Survivor.git
cd Tiny_Survivor
pip install -r requirements.txt
streamlit run src/app.py
```
And when you enter the game, select an agent and reward type in the sidebar, then press **Run Episode**. Now you can start to explore!

## Video Links
[Demo Video](https://youtu.be/qS1BVyBNlTg)

[Technical Walkthrough](https://youtu.be/3n-Yp-Eroqs)

## Evaluation
### Quantitative Results

All agents were evaluated over 500 greedy episodes (no exploration) after training in the ablation study.

The ablation study runs 4 experiments, which are Q-Learning + Sparse, Q-Learning + Shaped, DQN + Sparse, and DQN + Shaped.

| | Sparse Reward | Shaped Reward |
|---|---|---|
| **Q-Learning — avg reward** | -13.52 | -4.55 |
| **Q-Learning — win rate** | 52.2% | 58.4% |
| **Q-Learning — avg steps** | 35.91 | 32.31 |
| **DQN — avg reward** | -32.00 | -15.41 |
| **DQN — win rate** | 31.8% | 44.8% |
| **DQN — avg steps** | 45.47 | 40.68 |

---

**Q-Learning outperforms DQN on this task.** With sparse reward, Q-Learning achieves 52.2% win rate, while DQN's 31.8%. The gap narrows with shaped reward (58.4% vs 44.8%), but Q-Learning always wins. DQN is designed for more complex tasks.

**Reward shaping helps both agents.** Shaped reward improved Q-Learning win rate by 6.2 percentage points and DQN by 13 points. The effect is larger for DQN, likely because the distance bonus provides clearer guidance during the unstable early training phase. But the effect is not so significance for both.

**DQN's sharp epsilon decaying.** DQN's win rate peaked at only about 77% at episode 5000 then dropped sharply as epsilon decayed. The checkpoint at episode 5000 was saved as the best model.

**Generalization gap in Q-Learning.** In simulation across 300 random map layouts, Q-Learning's win rate dropped to 62% compared to 94% during training. This gap is caused by local loops.

---

## Individual Contributions
Solo project.
