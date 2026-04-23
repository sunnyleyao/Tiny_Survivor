"""
hamster_env.py — Custom Hamster Gymnasium Environment

Map: NxN grid (default 5x5)
Hamster starts at top-left corner (0, 0).

Items (randomly placed each episode):
    seed   : +5 reward
    magic  : +10 reward
    tape   : -5 reward
    stack  : -5 reward

Termination:
    Win      — all seeds and magic collected
    Lose     — score drops to 0 or below
    Truncate — max_steps reached

Each step costs -1 (movement penalty).

Observation (flat float32 vector):
    [hamster_row, hamster_col,   # normalized to [0,1]
     seed_map (flattened),       # binary, 1 if seed present
     magic_map (flattened),      # binary, 1 if magic present
     trap_map (flattened),       # binary, 1 if tape or stack present
     score_norm]                 # score / 100, clipped to [0,1]

Action space: Discrete(4)
    0 = up, 1 = down, 2 = left, 3 = right
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

EMPTY = 0
SEED  = 1
MAGIC = 2
TAPE  = 3
STACK = 4

ITEM_REWARD = {
    SEED:  +5,
    MAGIC: +10,
    TAPE:  -5,
    STACK: -5,
}

# terminal render symbols (for debugging)
ITEM_SYMBOL = {
    EMPTY: " . ",
    SEED:  " S ",
    MAGIC: " M ",
    TAPE:  " T ",
    STACK: " X ",
}


class HamsterEnv(gym.Env):
    """
    Custom Hamster grid-world Gymnasium environment.

    Parameters
    grid_size : int
        Side length of the square grid. Default 5.
    shaped_reward : bool
        If True, add a small bonus inversely proportional to distance
        to the nearest seed/magic. Used in ablation experiments.
    max_steps : int
        Maximum steps before episode is truncated.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, grid_size=5, shaped_reward=False, max_steps=200):
        super().__init__()

        self.grid_size     = grid_size
        self.shaped_reward = shaped_reward
        self.max_steps     = max_steps

        n = grid_size * grid_size

        # obs: [row, col] + seed_map + magic_map + trap_map + [score_norm]
        obs_dim = 2 + n + n + n + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # 0=up  1=down  2=left  3=right
        self.action_space = spaces.Discrete(4)

        # placeholders — populated in reset()
        self.grid        = None
        self.hamster_pos = None
        self.score       = None
        self.steps       = None
        self.items_left  = None


    def _random_pos(self, exclude: set):

        while True:
            r = int(self.np_random.integers(0, self.grid_size))
            c = int(self.np_random.integers(0, self.grid_size))
            if (r, c) not in exclude:
                return r, c

    def _place_items(self):
        """Randomly scatter items on the grid for this episode."""
        n_seeds  = 3
        n_magic  = 1
        n_tape   = 1
        n_stack  = 1

        taken = {(0, 0)}   # start position

        for _ in range(n_seeds):
            r, c = self._random_pos(taken)
            self.grid[r, c] = SEED
            taken.add((r, c))

        for _ in range(n_magic):
            r, c = self._random_pos(taken)
            self.grid[r, c] = MAGIC
            taken.add((r, c))

        for _ in range(n_tape):
            r, c = self._random_pos(taken)
            self.grid[r, c] = TAPE
            taken.add((r, c))

        for _ in range(n_stack):
            r, c = self._random_pos(taken)
            self.grid[r, c] = STACK
            taken.add((r, c))

        self.items_left = n_seeds + n_magic   # win when this hits 0

    def _get_obs(self):
        g = self.grid_size
        n = g * g
        r, c = self.hamster_pos

        seed_map  = np.zeros(n, dtype=np.float32)
        magic_map = np.zeros(n, dtype=np.float32)
        trap_map  = np.zeros(n, dtype=np.float32)

        for row in range(g):
            for col in range(g):
                idx  = row * g + col
                cell = self.grid[row, col]
                if cell == SEED:
                    seed_map[idx]  = 1.0
                elif cell == MAGIC:
                    magic_map[idx] = 1.0
                elif cell in (TAPE, STACK):
                    trap_map[idx]  = 1.0

        obs = np.concatenate([
            [r / max(g - 1, 1),
             c / max(g - 1, 1)],
            seed_map,
            magic_map,
            trap_map,
            [np.clip(self.score / 100.0, 0.0, 1.0)],
        ]).astype(np.float32)

        return obs

    def _get_info(self):
        return {
            "score":      self.score,
            "steps":      self.steps,
            "items_left": self.items_left,
            "pos":        self.hamster_pos,
        }

    def _nearest_goal_dist(self):
        """Manhattan distance to nearest uncollected seed or magic."""
        r, c  = self.hamster_pos
        min_d = float("inf")
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row, col] in (SEED, MAGIC):
                    min_d = min(min_d, abs(r - row) + abs(c - col))
        return min_d if min_d != float("inf") else 0

    # Gymnasium API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.hamster_pos = (0, 0)
        self.score = 50      # give agent enough steps to find rewards before dying
        self.steps = 0

        self._place_items()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        r, c = self.hamster_pos
        g = self.grid_size

        # movement (wall blocking — stay in place if out of bounds)
        dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        nr, nc = r + dr, c + dc
        if 0 <= nr < g and 0 <= nc < g:
            self.hamster_pos = (nr, nc)

        # step penalty
        self.steps += 1
        self.score -= 1
        reward      = -1.0

        # item interaction
        cell = self.grid[self.hamster_pos]
        if cell != EMPTY:
            item_r       = ITEM_REWARD[cell]
            reward      += item_r
            self.score  += item_r
            if cell in (SEED, MAGIC):
                self.items_left -= 1
            self.grid[self.hamster_pos] = EMPTY   # consume item

        # shaped reward (ablation variant)
        if self.shaped_reward and self.items_left > 0:
            d      = self._nearest_goal_dist()
            reward += 0.5 / (d + 1)

        # termination
        win       = self.items_left == 0
        lose      = self.score <= 0
        done      = win or lose
        truncated = self.steps >= self.max_steps

        if win:
            reward += 10.0   # win bonus

        info = self._get_info()
        info["win"]  = win
        info["lose"] = lose

        return self._get_obs(), reward, done, truncated, info

    # ── render (terminal only) ────────────────────────────────────────────────

    def render(self):
        """Print ASCII map to terminal. Call manually when debugging."""
        g   = self.grid_size
        sep = "+" + ("---+" * g)
        print(sep)
        for row in range(g):
            row_str = "|"
            for col in range(g):
                if (row, col) == self.hamster_pos:
                    row_str += " H |"
                else:
                    row_str += ITEM_SYMBOL.get(self.grid[row, col], " . ") + "|"
            print(row_str)
            print(sep)
        print(f"Score: {self.score}  |  Steps: {self.steps}  |  Items left: {self.items_left}\n")

    def get_state(self):
        """
        Return a human-readable dict of the current state.
        Useful for debugging and building your own custom visualizations.
        """
        return {
            "grid":        self.grid.copy(),
            "hamster_pos": self.hamster_pos,
            "score":       self.score,
            "steps":       self.steps,
            "items_left":  self.items_left,
        }

    def close(self):
        pass

