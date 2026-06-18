# tuned_qlearning_agent.py
# Drop-in replacement for QLearningAgent with Double Q-learning, eligibility traces,
# epsilon decay, and optimistic initialization.

from dataclasses import dataclass
import numpy as np

@dataclass
class TQLConfig:
    n_states: int = 16
    n_actions: int = 6
    alpha_start: float = 0.25      # initial learning rate
    alpha_min: float = 0.05        # floor for learning rate
    alpha_decay: float = 0.999     # per-step multiplicative decay

    gamma: float = 0.98            # stronger focus on long-term returns

    epsilon_start: float = 0.20    # start more exploratory
    epsilon_min: float = 0.02      # keep tiny exploration
    epsilon_decay: float = 0.9995  # slow decay works well here

    lam: float = 0.85              # Watkins Q(λ) eligibility traces

    optimistic_init: float = 2.0   # optimistic Q0 helps drive exploration
    clip_td: float = 10.0          # clamp TD errors to stabilize updates

    seed: int = 42                 # reproducibility

class TunedQLearningAgent:
    """
    Double Q-learning + Watkins Q(λ) with epsilon/alpha decay & optimistic init.
    API is compatible with the original QLearningAgent:
      - select_action(state: int) -> int
      - update(s, a, r_eff, s_next) -> None
      - recommend_topk(state: int, k: int = 3) -> list[int]
    """
    def __init__(self, cfg: TQLConfig = TQLConfig()):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Double Q tables
        self.Q1 = np.full((cfg.n_states, cfg.n_actions), cfg.optimistic_init, dtype=float)
        self.Q2 = np.full((cfg.n_states, cfg.n_actions), cfg.optimistic_init, dtype=float)

        # Eligibility traces (one per table)
        self.E1 = np.zeros_like(self.Q1)
        self.E2 = np.zeros_like(self.Q2)

        # Schedules
        self.alpha = cfg.alpha_start
        self.epsilon = cfg.epsilon_start

    # ------------- Policy -------------
    def _Qmean(self):
        # Target policy uses mean ensemble to reduce overestimation
        return (self.Q1 + self.Q2) * 0.5

    def select_action(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.cfg.n_actions))
        q = self._Qmean()[state]
        # break ties randomly to avoid bias
        return int(self.rng.choice(np.flatnonzero(q == q.max())))

    # ------------- Learning -------------
    def _decay_schedules(self):
        self.alpha = max(self.cfg.alpha_min, self.alpha * self.cfg.alpha_decay)
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def update(self, s: int, a: int, r_eff: float, s_next: int):
        """
        Watkins Q(λ) with Double Q-learning:
          - randomize which table is the 'online' estimator this step (Q1 or Q2)
          - traces accumulate on the online table only
          - trace reset (cut) on non-greedy actions as per Watkins
        """
        use_Q1 = bool(self.rng.integers(2))
        Qa, Qb = (self.Q1, self.Q2) if use_Q1 else (self.Q2, self.Q1)
        Ea, Eb = (self.E1, self.E2) if use_Q1 else (self.E2, self.E1)

        # 1) Greedy action under the *online* table for s_next (used for target)
        a_star = int(np.argmax(Qa[s_next]))

        # 2) Double Q target: bootstrap using the *other* table’s estimate for a*
        target = r_eff + self.cfg.gamma * Qb[s_next, a_star]

        # 3) TD error
        td = target - Qa[s, a]
        if self.cfg.clip_td is not None:
            td = float(np.clip(td, -self.cfg.clip_td, self.cfg.clip_td))

        # 4) Eligibility trace update (Watkins):
        #    - increment trace for (s,a)
        #    - if the actual next action != greedy (under Qa), we 'cut' traces
        Ea *= self.cfg.gamma * self.cfg.lam
        Ea[s, a] += 1.0

        # Apply TD update to all state-actions via traces
        Qa += self.alpha * td * Ea

        # Watkins trace cutting: if a != a_star at s_next, zero the traces; else keep them
        # (implemented by caller if they pass next action; here we approximate using policy)
        # We approximate by cutting only when policy is sufficiently exploratory:
        if self.epsilon > 0.05:
            Ea *= 0.0  # cut aggressively early in training for stability

        # Decay schedules
        self._decay_schedules()

    # ------------- Inference helpers -------------
    def recommend_topk(self, state: int, k: int = 3):
        q = self._Qmean()[state]
        order = np.argsort(-q)
        return [int(i) for i in order[:k]]

# ---------------- Small self-test / example ----------------
if __name__ == "__main__":
    # Tiny smoke-test to show the API works.
    cfg = TQLConfig(n_states=16, n_actions=6)
    agent = TunedQLearningAgent(cfg)
    s = 0
    for t in range(1000):
        a = agent.select_action(s)
        # Fake reward: prefer action 2 at all states
        r = 1.0 if a == 2 else -0.1
        s_next = (s + 1) % cfg.n_states
        agent.update(s, a, r, s_next)
        s = s_next
    print("Top-3 actions at state 0:", agent.recommend_topk(0, k=3))