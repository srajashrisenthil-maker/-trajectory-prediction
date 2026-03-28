# Intent & Trajectory Prediction for Autonomous Driving

A deep learning system that predicts the future movement of pedestrians and cyclists in autonomous driving scenarios.

## What it does
- Predicts the next 3 seconds of movement using the past 2 seconds of data
- Detects intent: straight, turning, slowing
- Calculates risk level: Low, Medium, High
- Simulates what happens when a virtual obstacle is added
- Shows 3 possible future paths with probabilities

## Model Architecture
- GRU Encoder — learns from past trajectory
- Social Attention Layer — considers nearby agents
- Multi-Modal Decoder — predicts 3 possible future paths
- Intent Detection Module — classifies behavior
- Risk Score Module — estimates danger level
- What-If Simulation Engine — simulates obstacle avoidance

## How to run

### Install dependencies
pip install numpy matplotlib scikit-learn torch pandas

### Run the project
python main.py

## Output
- prediction.png — predicted trajectories with risk zone
- whatif.png — before and after obstacle simulation

## Evaluation Metrics
- ADE (Average Displacement Error)
- FDE (Final Displacement Error)
- minADE
- minFDE

## Built with
- Python
- PyTorch
- NumPy
- Matplotlib
