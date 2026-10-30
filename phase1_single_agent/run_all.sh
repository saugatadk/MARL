#!/usr/bin/env bash
# ==============================================================================
# run_all.sh
# Master Execution Pipeline for Phase 1: Single-Agent RL Traffic Signal Control
# IOE Pashchimanchal Campus, Pokhara, Nepal
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="/home/anonymous/Major-Project/venv/bin/python3"

if [[ ! -f "$VENV_PYTHON" ]]; then
    echo "Virtual environment not found at $VENV_PYTHON. Using system python3."
    VENV_PYTHON="python3"
fi

export SUMO_HOME="${SUMO_HOME:-/usr/share/sumo}"

echo "========================================================================"
echo " Starting Full End-to-End Pipeline: IOE Pashchimanchal Campus Traffic"
echo " Python:    $VENV_PYTHON"
echo " SUMO_HOME: $SUMO_HOME"
echo " Time:      $(date)"
echo "========================================================================"

# Step 1: Build the SUMO network from OSM data
echo ""
echo ">>> [STAGE 1/5] Building Road Network & Generating Mixed Demand..."
"$SCRIPT_DIR/scripts/build_network.sh"

# Step 2: Run Classical Baselines
echo ""
echo ">>> [STAGE 2/5] Running Classical Baselines (Fixed-Time & Actuated)..."
"$VENV_PYTHON" "$SCRIPT_DIR/baselines/fixed_time.py" --sim-time 3600
"$VENV_PYTHON" "$SCRIPT_DIR/baselines/actuated.py" --sim-time 3600

# Step 3: Train DQN Agent
echo ""
echo ">>> [STAGE 3/5] Training Single-Agent DQN Policy (10,000 timesteps)..."
"$VENV_PYTHON" "$SCRIPT_DIR/training/train_dqn.py" --timesteps 10000 --seed 42

# Step 4: Train PPO Agent
echo ""
echo ">>> [STAGE 4/5] Training Single-Agent PPO Policy (10,000 timesteps)..."
"$VENV_PYTHON" "$SCRIPT_DIR/training/train_ppo.py" --timesteps 10000 --seed 42

# Step 5: Comparative Evaluation Across 5 Seeds
echo ""
echo ">>> [STAGE 5/5] Running Comparative 5-Seed Evaluation Benchmark..."
"$VENV_PYTHON" "$SCRIPT_DIR/training/evaluate.py" \
    --sim-time 3600 \
    --seeds 42 43 44 45 46 \
    --output-md "$SCRIPT_DIR/evaluation_results.md"

echo ""
echo "========================================================================"
echo " PIPELINE EXECUTION COMPLETE!"
echo " Checkpoints: $SCRIPT_DIR/training/checkpoints/"
echo " TensorBoard: $SCRIPT_DIR/training/logs/"
echo " Benchmark:   $SCRIPT_DIR/evaluation_results.md"
echo "========================================================================"
echo ""
echo "To launch the visual SUMO-GUI replay of the trained model, run:"
echo "  $VENV_PYTHON $SCRIPT_DIR/scripts/run_visual_demo.py --controller dqn --delay 50"
echo "========================================================================"
