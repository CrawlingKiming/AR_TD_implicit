# Implicit Updates for Average-Reward Temporal Difference Learning

This repository provides the source code necessary to conduct numerical experiments described in the paper.

## Setup

We recommend using a virtual environment to ensure reproducibility. You can create and activate one as follows:

```bash
# Create a virtual environment named .venv
python3 -m venv .venv

# Activate the environment
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

## Evaluation Experiement Examples 

For the evaluation experiments, we provide two examples: Markovian reward process and Boyan chain.
Each can be run with different step-size schedules:

```bash
python different_TD_fixed_points_im2.py --env MRP  --step_size_schedule constant  
python different_TD_fixed_points_im2.py --env Boyan  --step_size_schedule s_decay
```

## Control Experiment Examples 

For the control experiments, we provide two examples: Access-control queuing and Pendulum.
For example:

```bash
python control_experiment.py --env pendulum --num_experiments 30 --num_episodes 25000 
python control_experiment.py --env access_control --num_experiments 30 --num_episodes 25000 
```

