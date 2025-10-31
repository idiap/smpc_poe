# Project: Sampling-Based Motion Planning with Products of Experts (PoE)

This repository contains a minimal Isaac Gym-based example accompanying the paper “Sampling-Based Constrained Motion Planning with Products of Experts” by Razmjoo et al. It demonstrates pushing a Mustard bottle using MPPI-style sampling modified with PoE-inspired sampling mechanism in simulation.

The example focuses on a simple 2D pushing task (“Mustard Pushing”) with URDF assets under `URDF/` and runnable scripts under `Mustard Pushing/`.

**Key scripts**
- `Mustard Pushing/MPPI_pushing_isaac_planner.py` — launches a Zerorpc server that exposes the planner/simulator API.
- `Mustard Pushing/MPPI_pushing_isaac.py` — client that connects to the planner, sets cost terms, and executes the pushing routine with visualization.
- `Mustard Pushing/nvidia_wrapper.py` — Isaac Gym wrapper and utilities used by both.

## Prerequisites
- NVIDIA Isaac Gym.
- NVIDIA GPU with CUDA and appropriate drivers.
- Python 3.8 recommended.

## Environment
- A portable micromamba/conda environment is provided as `environment.yml` (exported from a working setup `tt_poe`).
- Create and activate it:
  
  micromamba env create -f environment.yml
  micromamba activate tt_poe

- Alternatively, use `requirements.txt` with pip after installing a CUDA-enabled PyTorch matching your drivers.

## Assets
- URDFs are in `URDF/` and include the Mustard bottle and simple shapes used by the example.
- Required precomputed data (generated offline) in the project root:
  - `factor_0.npy` … `factor_6.npy` — generate using `Mustard Pushing/Effective_dist_calc_tt.py` (see script for parameters and outputs).
  - `mustared_aligned_points.npy` — generate using the notebook `Mustard Pushing/create_aligned_points.ipynb`.
  Place the resulting `.npy` files in the repository root so `Mustard Pushing/nvidia_wrapper.py` can load them. Note the expected filename is `mustared_aligned_points.npy` (matching the current code). 

## Quick start
1) Install Isaac Gym locally. Download from https://developer.nvidia.com/isaac-gym/download and follow the instruction to install it.
2) Create an isolated environment (micromamba/conda or venv).
4) Export environment variables so Python and the dynamic loader can find Isaac Gym.
```
export ISAAC_GYM_ROOT=<path/to/Isaac/Gym/Env>

# let the dynamic loader find libpython and Isaac Gym's .so files
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$ISAAC_GYM_ROOT/isaacgym/python/isaacgym/_bindings/linux-x86_64:${LD_LIBRARY_PATH}"

# also ensure Python can find Isaac Gym's Python packages
export PYTHONPATH="$ISAAC_GYM_ROOT/isaacgym/python:${PYTHONPATH}"
```
5) Start the planner server, then run the client.


## Running the example
1) Terminal A — start the planner server:
  
  `python "Mustard Pushing/MPPI_pushing_isaac_planner.py" --num-horizon 20 --num-envs 64`

   - Binds Zerorpc at `tcp://0.0.0.0:4242`.
   - Runs headless Isaac Gym and exposes planning/simulation functions.

2) Terminal B — launch the client/visualization:
  
```
python "Mustard Pushing/MPPI_pushing_isaac.py" \
    --vis True \
    --mppi-noise 0.5 \
    --mppi-temp 0.05 \
    --state-cost 10.0 \
    --terminal-cost 1000.0 \
    --action-cost 0.001 \
    --sdf-cost 0.0 \
    --collision-cost 100000.0 \
    --exploration-cost 1.0 \
    --terminal-cond 0.1 \
    --ttgo-exploration 0.0
```
   - The defaults generally work; tune costs as desired.
   - The client connects to `127.0.0.1:4242`, sets cost terms, and steps the sim.

## Citation
If you use this code in your research, please cite:
```
@article{Razmjoo25IJRR,
	author={Razmjoo, A. and Xue, T. and Shetty, S. and Calinon, S.},
	title={Sampling-Based Constrained Motion Planning with Products of Experts},
	journal={International Journal of Robotics Research ({IJRR})},
	year={2025}
}
```
## License
See `LICENSE` for details.

## Third-Party Code
- This project includes code adapted from TTGO (GPLv3) by Idiap Research Institute in `Mustard Pushing/tt_utils_ttgo.py`. See `THIRD_PARTY_NOTICES.md` for details.
