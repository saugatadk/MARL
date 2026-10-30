# Phase 1: Single-Agent RL Traffic Signal Control on Real Road Network of IOE Pashchimanchal Campus, Pokhara

[![SUMO](https://img.shields.io/badge/SUMO-1.28.0-blue.svg)](https://sumo.dlr.de)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0+-green.svg)](https://gymnasium.farama.org)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.0+-orange.svg)](https://stable-baselines3.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11%20|%203.12-blue.svg)](https://python.org)

## 1. Project Overview & Context
This repository contains **Phase 1** of an undergraduate major project on **Adaptive Traffic Signal Control using Reinforcement Learning**. Before expanding to multi-agent distributed networks across the Pokhara valley in Phase 2, Phase 1 establishes a validated single-agent reinforcement learning pipeline on the real-world road network surrounding **IOE Pashchimanchal Campus (Western Regional Campus - WRC), Lamachaur, Pokhara, Nepal**.

### Key Highlights
- **Geographically Calibrated:** Ingests real OpenStreetMap data centered on IOE Pashchimanchal Campus and the Lamachaur Chowk / Baglung-Pokhara Highway corridor.
- **Mixed Traffic Profile:** Simulates realistic South Asian heterogeneous traffic conditions: **50% motorcycles/scooters**, **40% passenger cars/taxis**, and **10% buses/microbuses** (~700 veh/hr).
- **Standards-Compliant Interfaces:** Strict adherence to modern `Gymnasium` (Farama Foundation) and `Stable-Baselines3` standards.
- **TraCI Bottleneck Solution:** Operates on a discrete $\Delta t = 5\text{s}$ decision interval with automated $3\text{s}$ yellow safety clearance transitions.
- **Rigorous Benchmarking:** Evaluates two classical baselines (**Fixed-Time** and **Actuated Rule-Based**) alongside **DQN** and **PPO** across multiple random seeds.
- **Visual Validation:** Interactive `sumo-gui` visual replay module running at 50ms per step.

---

## 2. Geographical Location & Network Configuration
- **Location:** IOE Pashchimanchal Campus & Lamachaur Chowk, Pokhara, Kaski, Gandaki Province, Nepal.
- **Bounding Box (WGS84 Coordinates):**
  - Min Longitude (West): `83.9680`
  - Min Latitude (South): `28.2480`
  - Max Longitude (East): `83.9820`
  - Max Latitude (North): `28.2620`
- **Primary Signalized Junction:** Lamachaur Chowk (`Junction ID: 3700742618`), a 4-way intersection connecting North-South Lamachaur Marg (leading past the campus gates) and East-West corridors.
- **4-Phase Scheme:**
  - **Phase 0:** North-South Green (`rrrrGGGGrrrrrGGGGG`), Duration: 27s (Min: 10s, Max: 60s)
  - **Phase 1:** North-South Yellow (`rrrryyyyrrrrryyyyy`), Duration: 3s
  - **Phase 2:** East-West Green (`GGGGrrrrGGGGGrrrrr`), Duration: 27s (Min: 10s, Max: 60s)
  - **Phase 3:** East-West Yellow (`yyyyrrrryyyyyrrrrr`), Duration: 3s

---

## 3. Repository Structure
```
phase1_single_agent/
├── data/
│   ├── paschimanchal.osm         # Downloaded OSM map data for Lamachaur / WRC
│   ├── paschimanchal.net.xml     # SUMO road network generated via netconvert
│   ├── paschimanchal.poly.xml    # Campus buildings, grounds, and polygons
│   ├── typemap.xml               # OSM highway and road type mapping
│   ├── routes.rou.xml            # Generated 50/40/10 mixed vehicle routes
│   ├── simulation.sumocfg        # Master SUMO configuration
│   └── gui-settings.cfg          # GUI viewpoint and color settings
├── env/
│   ├── __init__.py
│   ├── sumo_env.py               # Custom Gymnasium environment wrapping TraCI
│   └── observation_extractor.py  # TraCI lane queue, wait time, and metric extractors
├── baselines/
│   ├── __init__.py
│   ├── fixed_time.py             # Static 60s cycle controller (27s G + 3s Y)
│   └── actuated.py               # Sensor-based actuated dynamic extension controller
├── training/
│   ├── __init__.py
│   ├── callbacks.py              # TensorBoard callback for domain traffic metrics
│   ├── train_dqn.py              # Stable-Baselines3 DQN training script
│   ├── train_ppo.py              # Stable-Baselines3 PPO training script
│   ├── evaluate.py               # 5-seed comparative benchmarking script
│   ├── checkpoints/              # Saved model weights (.zip)
│   └── logs/                     # TensorBoard scalar run logs
├── scripts/
│   ├── fetch_osm_map.py          # Resilient Overpass / OSM API downloader
│   ├── configure_tls.py          # Injects 4-phase signal logic into net.xml
│   ├── generate_traffic.py       # Injects Pokhara mixed vehicle distribution
│   ├── build_network.sh          # End-to-end automated build script
│   └── run_visual_demo.py        # Live SUMO-GUI replay launcher
├── requirements.txt
└── README.md
```

---

## 4. Reinforcement Learning Environment Design

### Action Space
`Discrete(2)`:
- `0`: Active Phase 0 — North-South Green (Corridor along WRC campus)
- `1`: Active Phase 2 — East-West Green

If the agent requests a phase switch, the environment automatically inserts a mandatory 3-second yellow clearance phase before activating the opposing green phase, guaranteeing pedestrian and vehicular safety.

### Observation Space
`Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)`:
- **Normalized Queue Length per Lane (6 dimensions):** Halting vehicles ($\text{speed} < 0.1\,\text{m/s}$) divided by maximum lane capacity.
- **Normalized Waiting Time per Lane (6 dimensions):** Accumulated seconds vehicles have been stationary, normalized by $300\,\text{s}$.
- **Active Green Phase One-Hot (2 dimensions):** `[1.0, 0.0]` for Phase 0 (NS) vs `[0.0, 1.0]` for Phase 2 (EW).
- **Normalized Elapsed Phase Time (1 dimension):** Seconds elapsed in current green phase divided by $\text{max\_green} = 60\,\text{s}$.

### Reward Function
$$r_t = - \sum_{l \in L} \left( w_l(t) + \alpha \cdot q_l(t) \right)$$
where $L$ is the set of all incoming intersection lanes, $w_l(t)$ is total accumulated waiting time on lane $l$, $q_l(t)$ is the queue length, and $\alpha = 0.5$. The reward directly penalizes congestion delay and queue accumulation.

---

## 5. Quick Start & Execution Guide

### 1. Prerequisites & Virtual Environment
Ensure Eclipse SUMO 1.18+ is installed on your Linux system:
```bash
sudo apt update && sudo apt install -y sumo sumo-tools sumo-gui
```

Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r phase1_single_agent/requirements.txt
```

### 2. Build the Network & Generate Traffic Demand
Execute the automated network build script:
```bash
cd phase1_single_agent
./scripts/build_network.sh
```
This script automatically:
1. Downloads OSM map data for the IOE Pashchimanchal Campus bounding box.
2. Converts the map into `paschimanchal.net.xml` via `netconvert`.
3. Injects the 4-phase traffic light logic at Lamachaur Chowk.
4. Generates campus building and terrain polygons via `polyconvert`.
5. Creates mixed vehicle routes (50% motorcycle, 40% passenger, 10% bus) for 1 hour.

### 3. Run Classical Baselines
To test the Fixed-Time static cycle:
```bash
python baselines/fixed_time.py --sim-time 3600
```
To test the Actuated sensor-based controller:
```bash
python baselines/actuated.py --sim-time 3600
```

### 4. Train Reinforcement Learning Agents
Train the **DQN** agent:
```bash
python training/train_dqn.py --timesteps 50000 --seed 42
```

Train the **PPO** agent:
```bash
python training/train_ppo.py --timesteps 50000 --seed 42
```

To monitor training curves in TensorBoard:
```bash
tensorboard --logdir training/logs
```
Logged metrics include:
- `rollout/ep_rew_mean`
- `custom/avg_waiting_time`
- `custom/avg_queue_length`
- `custom/throughput`

### 5. Multi-Seed Benchmarking & Comparative Evaluation
Run all 4 controllers across 5 independent random seeds:
```bash
python training/evaluate.py --sim-time 3600 --seeds 42 43 44 45 46
```

### 6. Live Visual Replay in SUMO-GUI
Watch the trained RL policy adapt signals to real-time traffic:
```bash
python scripts/run_visual_demo.py --controller dqn --delay 50
```
*(Options for `--controller`: `dqn`, `ppo`, `fixed`, `actuated`)*

---

## 6. Phase 2 Roadmap: Extension to Multi-Agent (MARL)
Phase 1 establishes the single-agent benchmark. Phase 2 extends this framework by:
1. Converting the single-agent environment into a PettingZoo `ParallelEnv`.
2. Modeling multiple adjacent intersections along Lamachaur Marg and Baglung-Pokhara Highway.
3. Training cooperative MARL policies (QMIX, MAPPO) to coordinate corridor green waves.
