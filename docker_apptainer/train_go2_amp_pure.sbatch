#!/bin/bash
#SBATCH --job-name=go2_amp_pure_amp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-pure-amp-%j.out
#SBATCH --error=slurm-pure-amp-%j.err

set -euo pipefail

# Expected layout:
#
# ~/AMP_docker_env/
# ├── legged_rl_gym.sif
# └── legged_rl_gym/
#
# Submit from ~/AMP_docker_env:
#   sbatch train_go2_amp_pure_amp.sbatch

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
SIF="${PROJECT_DIR}/legged_rl_gym.sif"
REPO="${PROJECT_DIR}/legged_rl_gym"

PYTHON="/opt/conda/envs/legged_rl/bin/python"
BRANCH="pure_amp"
TASK="go2_amp"
EXPERIMENT_NAME="go2_amp_example"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROS_DOMAIN_ID=100
export NVIDIA_VISIBLE_DEVICES=all
export NVIDIA_DRIVER_CAPABILITIES=all

if [ ! -f "${SIF}" ]; then
    echo "ERROR: Missing SIF: ${SIF}"
    exit 1
fi

if [ ! -d "${REPO}" ]; then
    echo "ERROR: Missing repo directory: ${REPO}"
    exit 1
fi

cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-no_slurm_job_id}"
echo "Node: $(hostname)"
echo "Project dir: ${PROJECT_DIR}"
echo "SIF: ${SIF}"
echo "Repo: ${REPO}"
echo "Branch: ${BRANCH}"
echo "Task: ${TASK}"

echo
echo "=== Checking repo branch ==="

if [ -d "${REPO}/.git" ]; then
    git -C "${REPO}" checkout "${BRANCH}"

    CURRENT_BRANCH="$(git -C "${REPO}" rev-parse --abbrev-ref HEAD)"
    if [ "${CURRENT_BRANCH}" != "${BRANCH}" ]; then
        echo "ERROR: repo is on branch ${CURRENT_BRANCH}, expected ${BRANCH}"
        exit 1
    fi
else
    echo "WARNING: ${REPO} is not a git checkout; cannot verify branch."
fi

echo
echo "=== GPU check ==="
nvidia-smi || true

echo
echo "=== Training GO2 AMP on pure_amp branch ==="
apptainer exec --nv \
    --bind "${REPO}:/root/legged_rl_gym" \
    --pwd /root/legged_rl_gym \
    "${SIF}" \
    "${PYTHON}" legged_gym/scripts/train.py \
        --task="${TASK}" \
        --headless \
        --num_envs=8192 \
        --max_iterations=50000

echo
echo "=== Exporting policy with play.py in headless mode ==="
apptainer exec --nv \
    --bind "${REPO}:/root/legged_rl_gym" \
    --pwd /root/legged_rl_gym \
    "${SIF}" \
    "${PYTHON}" legged_gym/scripts/play.py \
        --task="${TASK}" \
        --headless \
        --num_envs=1

echo
echo "=== Copying exported policy to deploy/exported_policy/go2/policy.pt ==="
mkdir -p "${REPO}/deploy/exported_policy/go2"

EXPORTED_POLICY="${REPO}/logs/${EXPERIMENT_NAME}/exported/policies/policy_1.pt"

if [ ! -f "${EXPORTED_POLICY}" ]; then
    echo "ERROR: Expected exported policy not found:"
    echo "  ${EXPORTED_POLICY}"
    echo
    echo "Available exported policies:"
    find "${REPO}/logs" -path "*/exported/policies/*.pt" -print || true
    exit 1
fi

cp -f "${EXPORTED_POLICY}" "${REPO}/deploy/exported_policy/go2/policy.pt"

echo "Policy exported to:"
echo "  ${REPO}/deploy/exported_policy/go2/policy.pt"

echo
echo "=== Recording policy video ==="
echo "record_policy.py uses the Isaac Gym viewer, so this step uses xvfb-run."
echo "Do not pass --headless to record_policy.py."

if command -v xvfb-run >/dev/null 2>&1; then
    xvfb-run -a -s "-screen 0 1280x720x24" \
        apptainer exec --nv \
            --bind "${REPO}:/root/legged_rl_gym" \
            --bind /tmp/.X11-unix:/tmp/.X11-unix \
            --pwd /root/legged_rl_gym \
            "${SIF}" \
            "${PYTHON}" legged_gym/scripts/record_policy.py \
                --task="${TASK}" \
                --num_envs=1

    echo
    echo "Video recording completed."
    echo "Check:"
    echo "  ${REPO}/record.mp4"
    echo "  ${REPO}/logs/${EXPERIMENT_NAME}/exported/frames/"
else
    echo
    echo "WARNING: xvfb-run was not found on this node."
    echo "Training and policy export completed, but video recording was skipped."
fi

echo
echo "=== Done ==="
