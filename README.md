# LineWorld Reinforcement Learning Pipeline

An end-to-end Reinforcement Learning (RL) prototype built with **Gymnasium**, **Stable-Baselines3**, **FastAPI**, and **Docker**. This project implements a custom 1D navigation environment (`LineWorld`), benchmarks a random agent against a trained Proximal Policy Optimization (`PPO`) policy, packages the training workflow for Docker, and exposes the environment over a REST API.

---

## Project Overview

* **Custom Environment (`env.py`):** A discrete 1D grid (`LineWorld`, default size `10`) where an agent starts at position `0` and chooses between moving left (`0`) or right (`1`) to reach the goal at `size - 1`.
  * **Reward Function:** `+1.0` upon reaching the terminal goal state; `-0.01` step penalty otherwise to incentivize shortest-path navigation.
* **Baseline & Validation:** Verifies Gymnasium API compliance with `check_env()` and benchmarks a random-action policy (`seed=0`).
* **RL Training (`train.py`):** Trains a Stable-Baselines3 `PPO` (`MlpPolicy`) agent for `20,000` timesteps and saves the trained artifact to `output/model`.
* **REST API (`server.py`):** Serves `LineWorld` via **FastAPI** and **Uvicorn** (`POST /reset` and `POST /step`) with an automated integration test script.
* **Containerization (`Dockerfile`):** Packages the environment and training script into a lightweight `python:3.11-slim` container.

---

## Repository Structure

```text
├── env.py                 # Custom Gymnasium LineWorld environment
├── train.py               # Environment check, random baseline, and PPO training
├── server.py              # FastAPI service exposing /reset and /step endpoints
├── requirements.txt       # Python dependencies (gymnasium, stable-baselines3, numpy, fastapi, uvicorn)
├── Dockerfile             # Container definition for reproducible training
└── ReinforcementLearning.ipynb  # Original Google Colab notebook
```

---

## Getting Started

### 1. Install Dependencies

```bash
pip install gymnasium==1.0.0 stable-baselines3 fastapi uvicorn requests
```

### 2. Run Training

Train the PPO agent for 20,000 timesteps and save the model weights to `output/model.zip`:

```bash
python train.py
```

### 3. Start the FastAPI Server

Launch the local environment API on port `8000`:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at `http://localhost:8000/docs`.

#### Example API Usage

```bash
# Reset the environment
curl -X POST http://localhost:8000/reset

# Step right (action=1)
curl -X POST "http://localhost:8000/step?action=1"
```

### 4. Run with Docker

Build and run the containerized training pipeline:

```bash
docker build -t lineworld-rl .
docker run --rm lineworld-rl
```

---

## Tech Stack

* **RL Framework:** Gymnasium (`1.0.0`), Stable-Baselines3 (`PPO`)
* **API & Serving:** FastAPI, Uvicorn, Requests
* **Packaging:** Python 3.11, Docker
