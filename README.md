# Stabilising a PINN for pigeon-flock interaction laws

CEE 292 course project. It reproduces the physics-informed neural network (PINN) of Kim et al. (2025),
*Commanding Emergent Behavior with Neural Networks*
([original code](https://github.com/dongjokim03/Collective_motions_training)). That network infers a pairwise interaction law from the GPS tracks of a ten-pigeon flock (Nagy et al. 2010).

The project then tests three changes, judged by how well the learned law works as a **forward model**, i.e. by simulating the flock for 140 s:
- a hard initial condition;
- self-adaptive loss weights;
- constraints on the learned force law.

## Repository layout

| Path | Contents |
|---|---|
| `training.ipynb` | Data, PINN, the four methods, the training loop, the check against the shipped results, and the loss curves |
| `simulation.ipynb` | Forward simulation of every learned law, the metrics, a summary table per method, and figures of the flight paths and metrics |
| `data/pigeon_flock_18980_to_19120.csv` | Flight of the 10 pigeons over 18980–19120 s, one row every 0.2 s (701 rows): time, then x, y, vx, vy of each bird A, B, C, D, G, H, I, J, L, M. Bird A is the leader. Training uses the first 101 rows (18980–19000 s). |
| `results/learned_coefficients.csv` | Our 40 trained runs (4 methods × seeds 0–9): final losses, v0, and the 25 learned coefficients |
| `results/training_logs.csv` | Every optimisation step of those 40 runs |
| `figures/` | Figures written by the notebooks |

The method keys used in the notebooks and CSVs are:

| Key | Method |
|---|---|
| `full` | Baseline |
| `hic` | Hard IC |
| `aw2` | Adaptive weights |
| `pull_tail_v0` | Constrained law |

## How to reproduce

```bash
pip install -r requirements.txt
jupyter notebook          # then open the notebooks from this folder
```

1. **`simulation.ipynb`** runs the shipped coefficients of all 40 runs (about 5–10 min on a laptop CPU) and rebuilds the summary table and the figures.
2. **`training.ipynb`** trains one demonstration run (the baseline, seed 3, about 5–10 min) and compares it with the shipped result. It also redraws the loss curves from the shipped logs.
   - To retrain everything, set `VARIANTS_TO_TRAIN = ['full', 'hic', 'aw2', 'pull_tail_v0']` and `SEEDS = range(10)`. That takes about 5–10 h on one CPU core.
   - Then set `RESULTS_DIR = Path("outputs")` in `simulation.ipynb` to simulate your own runs.

**Bit-exact reproducibility.** Training runs on the CPU with one thread. With the pinned versions (Python 3.12, PyTorch 2.14.1), a run gives exactly the shipped coefficients and losses. Other versions or hardware give tiny rounding differences, which grow during L-BFGS, so individual runs can end differently. Medians over 10 seeds should be similar.

## Changes from the original code

The physics, loss, network, hyperparameters, and early-stopping rule are those of the original notebook. A few bugs were fixed:
- **Adam.** The original re-created the Adam optimiser at every step, which discarded its momentum. Here it is created once per run.
- **NaN guard.** The original tested `isfinite(loss).item` without the `()`, which is always true, so a NaN loss was never caught.
- **Seeding.** PyTorch is seeded, so runs are repeatable.
- **Retries.** The original retrains from new random initialisations until a run meets fixed loss thresholds. Here, each seed is trained once, with no such test.

**Simulation, compared with the original simulation notebook:**
- one common centre for all birds;
- the measured initial velocities;
- the same cruising speed v0 as in training;
- no random noise in the equations of motion.

## Data and references

- Data: M. Nagy, Z. Ákos, D. Biro and T. Vicsek, *Hierarchical group dynamics in pigeon flocks*, Nature 464, 890–893 (2010). The CSV joins the two files of the original repository (18980–19000 s and 19000–19120 s).
- D. Kim et al., *Commanding Emergent Behavior with Neural Networks* (2025), code: https://github.com/dongjokim03/Collective_motions_training.
