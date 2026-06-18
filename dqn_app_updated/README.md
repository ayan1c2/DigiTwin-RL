# Digital Twin Flask/FastAPI App (DQN inference)

This update switches the recommender from a tabular Q-table to a **Deep Q-Network (DQN)** loaded from a pickle:
`models/best_dqn_agent.pkl`.

## Expected pickle formats (any one supported)

A) layers list:
```python
{
  "ACTIONS": ["sleep_early", ...],
  "state_dim": 26,
  "layers": [
    {"W": W1, "b": b1},
    {"W": W2, "b": b2}
  ]
}
```

B) weights list:
```python
{
  "ACTIONS": [...],
  "state_dim": 26,
  "weights": [(W1,b1), (W2,b2)]
}
```

C) explicit keys:
```python
{
  "ACTIONS": [...],
  "state_dim": 26,
  "W1": W1, "b1": b1, "W2": W2, "b2": b2
}
```

Where `W1.shape == (26, hidden)`, `b1.shape == (hidden,)`,
and `W2.shape == (hidden, len(ACTIONS))`, `b2.shape == (len(ACTIONS),)`.

## Run

From the project root (one level above `app/`):
```bash
uvicorn app.main:app --reload --port 5000
```
