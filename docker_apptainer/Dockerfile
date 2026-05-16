FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS torch-builder

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTORCH_TAG=v2.3.1
ARG MAX_JOBS=16

ENV CONDA_DIR=/opt/conda
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CONDA_DIR}/bin:${PATH}

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    pkg-config \
    ninja-build \
    cmake \
    libopenblas-dev \
    libomp-dev \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda config --system --set channel_priority strict && \
    conda create -y -n legged_rl python=3.8 pip "setuptools<70" wheel && \
    conda clean -afy

ENV CONDA_DEFAULT_ENV=legged_rl
ENV PATH=${CONDA_DIR}/envs/legged_rl/bin:${CONDA_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CONDA_DIR}/envs/legged_rl/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN python -m pip install --upgrade pip "setuptools<70" wheel && \
    python -m pip install --no-cache-dir \
    numpy==1.23.5 \
    pyyaml \
    typing_extensions \
    sympy \
    networkx \
    jinja2 \
    filelock \
    fsspec \
    packaging \
    psutil \
    future \
    six \
    requests \
    cmake==3.31.6 \
    ninja

RUN git clone --recursive --branch ${PYTORCH_TAG} https://github.com/pytorch/pytorch.git /opt/pytorch

WORKDIR /opt/pytorch

# Patch PyTorch 2.3.1 CUDA architecture parsing so two-digit major arch
# values like 12.0 / sm_120 are accepted.
RUN python - <<'PY'
from pathlib import Path

cmake_file = Path("cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake")
s = cmake_file.read_text()

old = r'if(arch_name MATCHES "^([0-9]\\.[0-9](\\([0-9]\\.[0-9]\\))?)$")'
new = r'if(arch_name MATCHES "^([0-9]+\\.[0-9](\\([0-9]+\\.[0-9]\\))?)$")'
if old not in s:
    raise SystemExit("Could not find CUDA arch regex in select_compute_arch.cmake")
s = s.replace(old, new)

marker = 'elseif(${arch_name} STREQUAL "Hopper")\n      set(arch_bin 9.0)\n      set(arch_ptx 9.0)\n'
if marker in s and 'STREQUAL "Blackwell"' not in s:
    s = s.replace(
        marker,
        marker + '    elseif(${arch_name} STREQUAL "Blackwell")\n      set(arch_bin 12.0)\n      set(arch_ptx 12.0)\n'
    )

cmake_file.write_text(s)

cpp_ext = Path("torch/utils/cpp_extension.py")
s = cpp_ext.read_text()

old = "'9.0', '9.0a']"
new = "'9.0', '9.0a', '12.0']"
if old not in s:
    raise SystemExit("Could not find supported_arches list in cpp_extension.py")
s = s.replace(old, new)

old = 'num = arch[0] + arch[2:].split("+")[0]'
new = 'num = arch.split("+")[0].replace(".", "")'
if old not in s:
    raise SystemExit("Could not find arch-to-number conversion in cpp_extension.py")
s = s.replace(old, new)

cpp_ext.write_text(s)
PY

# Single-GPU RL does not need NCCL/distributed PyTorch. Disabling distributed
# makes the source build substantially less fragile.
RUN export USE_CUDA=1 \
    USE_CUDNN=1 \
    USE_NCCL=0 \
    USE_DISTRIBUTED=0 \
    USE_MPI=0 \
    BUILD_TEST=0 \
    TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0+PTX" \
    MAX_JOBS=${MAX_JOBS} \
    CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" \
    CMAKE_PREFIX_PATH="$(python -c 'import sys; print(sys.prefix)')" \
    && python setup.py bdist_wheel


FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG ISAACGYM_TARBALL=IsaacGym_Preview_3_Package.tar.gz
ARG REPO_BRANCH=lpf

ENV CONDA_DIR=/opt/conda
ENV CUDA_HOME=/usr/local/cuda
ENV CONDA_DEFAULT_ENV=legged_rl
ENV PATH=${CONDA_DIR}/envs/legged_rl/bin:${CONDA_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CONDA_DIR}/envs/legged_rl/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
ENV QT_X11_NO_MITSHM=1
ENV TORCH_CUDA_ARCH_LIST=12.0+PTX
ENV PYTHONUNBUFFERED=1
ENV TERM=xterm-256color
ENV LEGGED_RL_GYM_ROOT_DIR=/root/legged_rl_gym
ENV LEGGED_RL_GYM_BRANCH=${REPO_BRANCH}

SHELL ["/bin/bash", "-lc"]

WORKDIR /root

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    git \
    wget \
    curl \
    ca-certificates \
    pkg-config \
    patchelf \
    ninja-build \
    xvfb \
    xauth \
    libopenblas0-pthread \
    libopenmpi-dev \
    libomp-dev \
    libgl1 \
    libglvnd0 \
    libglx0 \
    libegl1 \
    libosmesa6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libx11-6 \
    libxrender1 \
    libxi6 \
    libxrandr2 \
    libxcursor1 \
    libxinerama1 \
    libglfw3 \
    libglew2.2 \
    libvulkan1 \
    vulkan-tools \
    mesa-utils \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda config --system --set channel_priority strict && \
    conda create -y -n legged_rl python=3.8 pip "setuptools<70" wheel && \
    conda clean -afy

RUN python -m pip install --upgrade pip "setuptools<70" wheel && \
    python -m pip install --no-cache-dir \
    numpy==1.23.5 \
    gym==0.23.1 \
    matplotlib==3.7.5 \
    tensorboard==2.8.0 \
    pybullet==3.2.1 \
    opencv-python==4.5.5.64 \
    pillow \
    imageio \
    imageio-ffmpeg \
    pyyaml \
    scipy \
    tqdm \
    termcolor \
    moviepy==1.0.3 \
    mujoco==3.2.3 \
    protobuf==3.20.3 \
    sympy \
    networkx \
    jinja2 \
    filelock \
    fsspec \
    typing_extensions \
    packaging \
    psutil

COPY --from=torch-builder /opt/pytorch/dist/torch-*.whl /tmp/

RUN python -m pip install --no-cache-dir --no-deps /tmp/torch-*.whl && \
    rm -f /tmp/torch-*.whl

COPY ${ISAACGYM_TARBALL} /tmp/isaacgym.tar.gz

RUN tar -xzf /tmp/isaacgym.tar.gz -C /root && \
    rm /tmp/isaacgym.tar.gz && \
    test -d /root/isaacgym/python

WORKDIR /root/isaacgym/python

# Important: no deps. Isaac Gym's installer can otherwise pull an old torch.
RUN python -m pip install --no-cache-dir --no-deps -e .

COPY legged_rl_gym /root/legged_rl_gym

WORKDIR /root/legged_rl_gym/rsl_rl
RUN python -m pip install --no-cache-dir --no-deps -e .

WORKDIR /root/legged_rl_gym
RUN python -m pip install --no-cache-dir --no-deps -e .

RUN echo "source ${CONDA_DIR}/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate legged_rl" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "export TERM=xterm-256color" >> /root/.bashrc && \
    echo "export NVIDIA_VISIBLE_DEVICES=all" >> /root/.bashrc && \
    echo "export NVIDIA_DRIVER_CAPABILITIES=all" >> /root/.bashrc && \
    echo "export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json" >> /root/.bashrc && \
    echo "export QT_X11_NO_MITSHM=1" >> /root/.bashrc && \
    echo "export LEGGED_RL_GYM_ROOT_DIR=/root/legged_rl_gym" >> /root/.bashrc && \
    echo "export LEGGED_RL_GYM_BRANCH=${REPO_BRANCH}" >> /root/.bashrc && \
    echo "export LD_LIBRARY_PATH=${CONDA_DIR}/envs/legged_rl/lib:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH}" >> /root/.bashrc && \
    echo "" >> /root/.bashrc && \
    echo "alias ls='ls --color=auto'" >> /root/.bashrc && \
    echo "alias grep='grep --color=auto'" >> /root/.bashrc && \
    echo "alias diff='diff --color=auto'" >> /root/.bashrc && \
    echo "export PS1='\[\e[1;32m\]\u@\h\[\e[0m\]:\[\e[1;34m\]\w\[\e[0m\]\\$ '" >> /root/.bashrc

# Isaac Gym must be imported before torch.
RUN python - <<'PY'
from isaacgym import gymapi, gymtorch
import torch
import legged_gym
import rsl_rl
import mujoco

print("isaacgym import ok")
print("torch:", torch.__version__, "torch cuda:", torch.version.cuda)
print("legged_gym:", legged_gym.LEGGED_GYM_ROOT_DIR)
print("rsl_rl import ok")
print("mujoco:", mujoco.__version__)
PY

WORKDIR /root/legged_rl_gym

CMD ["/bin/bash"]
