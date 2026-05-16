# legged_rl_gym Docker and Slurm Usage

Docker and Apptainer setup for running [`RX-00/legged_rl_gym`](https://github.com/RX-00/legged_rl_gym) with Isaac Gym.

Supported branches:

- `lpf`
- `pure_amp`
---

## Expected Directory Layout

This directory should contain:

```bash
AMP_docker_env/
├── Dockerfile
├── Makefile
├── README.md
├── IsaacGym_Preview_3_Package.tar.gz
├── legged_rl_gym.sif
├── legged_rl_gym/
├── train_go2_amp_lpf.sbatch
└── train_go2_amp_pure.sbatch
```

If `legged_rl_gym/` does not exist, the Makefile will clone it automatically.

---

## Build the Docker Image

Build using the default branch, `lpf`:

```bash
make build
```

Or explicitly choose a branch:

```bash
make build BRANCH=lpf
make build BRANCH=pure_amp
```

---

## Run Docker [DON'T ACTUALLY DO THIS]

Run with display support, if available:

```bash
make run BRANCH=lpf
```

Run headless:

```bash
make run_headless BRANCH=lpf
```

Use the `pure_amp` branch instead:

```bash
make run_headless BRANCH=pure_amp
```

The Makefile automatically switches the local `legged_rl_gym` checkout to the requested branch before running.

---

## Test the Docker Environment

```bash
make test BRANCH=lpf
```

or:

```bash
make test BRANCH=pure_amp
```

This checks imports for Isaac Gym, PyTorch, `legged_gym`, `rsl_rl`, MuJoCo, and CUDA.

---

## Switch Branches

Switch to `lpf`:

```bash
make branch BRANCH=lpf
```

Switch to `pure_amp`:

```bash
make branch BRANCH=pure_amp
```

In most cases, switching branches does **not** require rebuilding the Docker image because the repo is mounted into the container at runtime.

Rebuild only if the branch changes dependencies or native build requirements.

---

## Attach to a Running Docker Container

```bash
make attach BRANCH=lpf
```

or:

```bash
make attach BRANCH=pure_amp
```

---

## Clean Docker Containers

```bash
make clean
```

This removes stopped or running `legged_rl_gym_lpf` and `legged_rl_gym_pure_amp` containers.

---

## Build the Apptainer Image

After building the Docker image, create the Apptainer image:

```bash
apptainer build legged_rl_gym.sif docker-daemon:legged_rl_gym:latest
```

If `/tmp` is too small, use a scratch directory:

```bash
mkdir -p /scratch/$USER/apptainer_tmp /scratch/$USER/apptainer_cache

export APPTAINER_TMPDIR=/scratch/$USER/apptainer_tmp
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer_cache

apptainer build legged_rl_gym.sif docker-daemon:legged_rl_gym:latest
```

---

## GO2 AMP Slurm Training

This folder contains two Slurm scripts for training GO2 AMP with Apptainer:

```bash
train_go2_amp_lpf.sbatch
train_go2_amp_pure.sbatch
```

Use `train_go2_amp_lpf.sbatch` for the `lpf` branch and `train_go2_amp_pure.sbatch` for the `pure_amp` branch.

---

## Before Submitting a Slurm Job

Make sure the repo is on the correct branch.

For `lpf`:

```bash
cd ~/AMP_docker_env
make branch BRANCH=lpf
```

For `pure_amp`:

```bash
cd ~/AMP_docker_env
make branch BRANCH=pure_amp
```

Make sure the Apptainer image exists:

```bash
ls legged_rl_gym.sif
```

---

## Submit Slurm Training Jobs

Train on the `lpf` branch:

```bash
sbatch train_go2_amp_lpf.sbatch
```

Train on the `pure_amp` branch:

```bash
sbatch train_go2_amp_pure.sbatch
```

Both scripts run:

```bash
python legged_gym/scripts/train.py \
  --task=go2_amp \
  --headless \
  --num_envs=8192 \
  --max_iterations=50000
```

---

## What the Slurm Scripts Do

Each script:

1. Checks that `legged_rl_gym.sif` exists.
2. Checks that `legged_rl_gym/` exists.
3. Verifies or switches to the intended branch.
4. Runs GO2 AMP training headlessly.
5. Exports the trained policy using `play.py --headless`.
6. Copies the exported policy to:

```bash
legged_rl_gym/deploy/exported_policy/go2/policy.pt
```

7. Attempts to record a video using `record_policy.py`.

Video recording requires a display. On headless Slurm nodes, the script uses `xvfb-run` if available. If `xvfb-run` is not available, training and policy export still complete, but video recording is skipped.

---

## Monitor Slurm Jobs

Check queue status:

```bash
squeue -u $USER
```

Follow output:

```bash
tail -f slurm-<JOBID>.out
```

For the pure AMP script:

```bash
tail -f slurm-pure-amp-<JOBID>.out
```

---

## Outputs

Training logs and checkpoints are saved under:

```bash
legged_rl_gym/logs/
```

Exported policy:

```bash
legged_rl_gym/deploy/exported_policy/go2/policy.pt
```

Recorded video, if successful:

```bash
legged_rl_gym/record.mp4
```

or frame/video outputs under:

```bash
legged_rl_gym/logs/go2_amp_example/exported/
```

---

## Notes

- Use `lpf` for the LPF branch workflow.
- Use `pure_amp` for the pure AMP branch workflow.
- Do not run `record_policy.py` with `--headless`; it needs a viewer.
- The training command itself is fully headless and Slurm-safe.
- The repo is bind-mounted into the container, so logs, checkpoints, exported policies, datasets, and videos persist on the host.
