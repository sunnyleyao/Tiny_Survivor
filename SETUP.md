

## Requirements
Python 3.11
pip

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sunnyleyao/rl-game.git
cd rl-game
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Web App

The trained models are already included in the repository (`q_table_sparse.pkl`, `q_table_shaped.pkl`, `dqn_sparse.pth`). You do not need to retrain anything to run the app.

```bash
streamlit run app.py
```

This will open the app locally. Select an agent (Q-Learning or DQN), choose a reward type(Sparse or Shaped), and press  **Run Episode** to watch the hamster play.

The app is also deployed publicly at: **https://rl-game-kc3wxzhc7fdrdhrghfw937.streamlit.app**
But the DQN mode is only available when running locally due to PyTorch compatibility issues with Streamlit Cloud.
## Re-training the Agents (Optional)

If you want to retrain from scratch:

```bash
# Train Q-Learning (both sparse and shaped reward, 50K episodes, ~10 min)
python q_learning.py

# Train DQN (sparse reward, 5K episodes, ~30 min on CPU)
python dqn.py
```

## Running the Analysis Notebook

After training (or using the provided model files), open the analysis notebook:

```bash
jupyter notebook analysis.ipynb
```

Run all cells in order to reproduce all plots and evaluation results.

## Running Evaluation Scripts

```bash
# Failure case analysis, behavioral analysis, simulation evaluation
python evaluate.py

# Learning curves, win rate plots, ablation study table
python compare.py
```

## File Structure

```
rl-game/
├── hamster_env.py        # Custom Gymnasium environment
├── q_learning.py         # Tabular Q-Learning agent
├── dqn.py                # DQN agent (PyTorch)
├── compare.py            # Learning curves + ablation study
├── evaluate.py           # Failure analysis + behavioral analysis
├── app.py                # Streamlit web app
├── analysis.ipynb        # Full analysis notebook
├── assets/               # Custom visual assets (hand-drawn)
├── q_table_sparse.pkl    # Trained Q-Learning model (sparse)
├── q_table_shaped.pkl    # Trained Q-Learning model (shaped)
├── dqn_sparse.pth        # Trained DQN model
├── requirements.txt
├── README.md
├── SETUP.md
└── ATTRIBUTION.md
```

## Notes

- No external APIs or secret keys required.
- All trained model files are included — no GPU needed to run the app.
- DQN requires `torch`. If you encounter import errors on Python 3.14+, use Python 3.11.