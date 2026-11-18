# Reinforcement Learning Projects

A collection of hands-on Reinforcement Learning (RL) projects built in Python with **Gymnasium** and **Stable-Baselines3**. These notebooks walk through core RL concepts—from basic environment loops and tabular Q-learning to training a PPO agent and serving it via a FastAPI web service.

---

## Repository Contents

```text
├── ReinforcementLearning.ipynb    # Custom LineWorld env, PPO training, Docker & FastAPI service
├── ReinforcementLearning_2.ipynb  # CartPole basics, Blackjack Q-Learning & custom 2D GridWorld
└── README.md
```

---

## Projects Overview

### 1. `LineWorld` RL & API Deployment (`ReinforcementLearning.ipynb`)
* **Custom 1D Environment (`env.py`):** A 10-position discrete track (`LineWorld`) where an agent starts at `0` and moves left (`0`) or right (`1`) to reach the goal at `9` (`+1.0` goal reward, `-0.01` step penalty).
* **Baseline & PPO Training:** Verifies the environment with `check_env()`, logs a random-policy baseline, and trains a **Stable-Baselines3** `PPO` (`MlpPolicy`) model for `20,000` timesteps.
* **API & Containerization (`server.py`, `Dockerfile`):** Exposes `POST /reset` and `POST /step` endpoints using **FastAPI** and **Uvicorn**, and packages the training setup for Docker.

### 2. RL Fundamentals & Custom GridWorld (`ReinforcementLearning_2.ipynb`)
* **CartPole Basics:** Demonstrates the standard Gymnasium `make` → `reset` → `step` interaction loop using random actions in `CartPole-v1`.
* **Blackjack Q-Learning:** Implements a tabular `BlackjackAgent` with an $\epsilon$-greedy policy, trains it across `100,000` hands of `Blackjack-v1`, plots moving-average reward/length/error curves with **Matplotlib**, and evaluates win rate over `1,000` greedy test episodes.
* **Custom 2D GridWorld:** Builds and registers a `5x5` `GridWorldEnv` (`gymnasium_env/GridWorld-v0`), validates it with `check_env()`, runs 4 parallel environments with `SyncVectorEnv`, and flattens observations using `FlattenObservation`.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install gymnasium==1.0.0 stable-baselines3 numpy matplotlib tqdm fastapi uvicorn requests
```

### 2. Run the Notebooks

Open `ReinforcementLearning.ipynb` or `ReinforcementLearning_2.ipynb` in **Google Colab** or **Jupyter Notebook** and run the cells from top to bottom.

### 3. Run the `LineWorld` FastAPI Server (Optional)

After running the first notebook to generate `env.py` and `server.py`, start the local API:

```bash
uvicorn server:app --port 8000
```

Interactive Swagger docs are available at `http://localhost:8000/docs`.

---

## Tech Stack

* **RL & Environments:** Gymnasium, Stable-Baselines3
* **Data & Visualization:** NumPy, Matplotlib, tqdm
* **API & Deployment:** FastAPI, Uvicorn, Docker
