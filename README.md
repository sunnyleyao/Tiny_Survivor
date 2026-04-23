# Tiny Survivor
This is a reinforcement learning game. The game happens in a custom grid-world where a hamster learns to collect food and avoid traps using two RL algorithms: tabular Q-Learning and Deep Q-Network (DQN). No rules were given to the agents, and they learned entirely through trial and error.

==## What is this project about==
Tiny Survivor is a 5×5 grid-world game built from a custom Gymnasium environment. A hamster named Toto starts in the top-left corner and must collect all seeds (+5) and a Helsius energy drink (+10) while avoiding a cat's paw (-5) and a bin (-5). The score starts at 50 which tracks both the hamster's health and its progress through the game. Every step costs 1 point, and the game ends when Toto collects everything or the score hits zero. Two RL agents were trained to play the game. The first is a tabular Q-Learning agent that learns a lookup table of optimal actions, and the other is a DQN agent that uses a neural network to approximate Q-values. And agents are trained under two reward designs which are sparse reward (only item pickups and traps) and shaped reward (with an additional bonus for moving toward the nearest goal). The project mainly compares these two approaches and investigates whether extra guidance helps either agent learn faster. It compares all four combinations and draws conclusions about which algorithm and reward design works best in this environment.

**Research Problem**: *In the same custom environment, does DQN actually learn better than tabular Q-learning, and does reward shaping help either of them?*

## Quick Start
Type the following in the terminal, and you will directly open the website application of this game and start to explore.
```bash
git clone https://github.com/sunnyleyao/Tiny_Survivor.git
cd Tiny_Survivor
pip install -r requirements.txt
streamlit run src/app.py
```

## Video Links
[Demo Video](https://youtu.be/RiIkUA4J3uA?si=VbDS1NYV3sfjZSj8)

**Technical Walkthrough:**

## Evaluation

## Individual Contributions
Solo project.
