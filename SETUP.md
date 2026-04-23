## Requirements
Python 3.11
pip
No external APIs required for my project.

## Installation

1. The first step is to clone the repository
```bash
git clone https://github.com/sunnyleyao/Tiny_Survivor.git
```

2. The next step is to install dependencies
```bash
pip install -r requirements.txt
```

3. Trained models are already included in the repository. Then you can directly run the functional web application.

```bash
streamlit run app.py
```

This will open the app locally. You can select an agent (Q-Learning or DQN), choose a reward type (Sparse or Shaped), and press  **Run Episode** to watch the hamster play.

4. And you can open the my analysis.

```bash
jupyter notebook analysis.ipynb
```

And you can run all cells in order to see all plots and evaluation results.

5. If you want to re-train the model, you can train in the terminal.

```bash
python q_learning.py   # ~10 min
python dqn.py          # ~30 min on CPU
```




