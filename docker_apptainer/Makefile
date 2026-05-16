IMAGE_NAME := legged_rl_gym
CONTAINER_NAME := legged_rl_gym_$(BRANCH)
DOMAIN_ID := 100

REPO_DIR := legged_rl_gym
REPO_URL := https://github.com/RX-00/legged_rl_gym.git

BRANCH ?= lpf
ISAACGYM_TARBALL ?= IsaacGym_Preview_3_Package.tar.gz
MAX_JOBS ?= 16

.PHONY: check_branch_arg repo checkout_branch branch build build_lpf build_pure_amp \
	run run_lpf run_pure_amp run_headless run_headless_lpf run_headless_pure_amp \
	attach test shell clean train_go1 train_go2 train_a1

check_branch_arg:
	@if [ "$(BRANCH)" != "lpf" ] && [ "$(BRANCH)" != "pure_amp" ]; then \
		echo "ERROR: BRANCH must be either 'lpf' or 'pure_amp'."; \
		echo "Example: make run BRANCH=lpf"; \
		echo "Example: make run BRANCH=pure_amp"; \
		exit 1; \
	fi

repo: check_branch_arg
	@if [ ! -d "$(REPO_DIR)" ]; then \
		echo "Cloning $(REPO_URL), branch $(BRANCH), into $(REPO_DIR)"; \
		git clone --branch $(BRANCH) --single-branch $(REPO_URL) $(REPO_DIR); \
	elif [ ! -d "$(REPO_DIR)/.git" ]; then \
		echo "ERROR: $(REPO_DIR) exists, but it is not a git checkout."; \
		echo "Move it aside or replace it with a clone of $(REPO_URL)."; \
		exit 1; \
	else \
		echo "$(REPO_DIR) already exists."; \
	fi

checkout_branch: repo
	@echo "Switching $(REPO_DIR) to branch $(BRANCH)"
	@cd $(REPO_DIR) && \
		git fetch origin $(BRANCH):refs/remotes/origin/$(BRANCH) && \
		if git show-ref --verify --quiet refs/heads/$(BRANCH); then \
			git checkout $(BRANCH); \
		else \
			git checkout -b $(BRANCH) origin/$(BRANCH); \
		fi && \
		git pull --ff-only origin $(BRANCH)

branch: checkout_branch
	@cd $(REPO_DIR) && git status --short --branch

build: checkout_branch
	docker build . -t $(IMAGE_NAME) \
		--build-arg ISAACGYM_TARBALL=$(ISAACGYM_TARBALL) \
		--build-arg MAX_JOBS=$(MAX_JOBS) \
		--build-arg REPO_BRANCH=$(BRANCH)

build_lpf:
	$(MAKE) build BRANCH=lpf

build_pure_amp:
	$(MAKE) build BRANCH=pure_amp

run: checkout_branch
	@if command -v xhost >/dev/null 2>&1; then xhost +local:docker; fi
	touch .bash_history
	docker run -it --rm --name $(CONTAINER_NAME) \
		--gpus all \
		--privileged \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--env="DISPLAY" \
		--env="QT_X11_NO_MITSHM=1" \
		--env="NVIDIA_VISIBLE_DEVICES=all" \
		--env="NVIDIA_DRIVER_CAPABILITIES=all" \
		--env="VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json" \
		--env="ROS_DOMAIN_ID=$(DOMAIN_ID)" \
		--env="LEGGED_RL_GYM_BRANCH=$(BRANCH)" \
		--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
		--volume="/home/$(USER)/.Xauthority:/root/.Xauthority:rw" \
		--volume="/usr/share/vulkan/icd.d:/usr/share/vulkan/icd.d:ro" \
		--volume="/dev/input:/dev/input:rw" \
		--volume="$(CURDIR)/$(REPO_DIR):/root/legged_rl_gym" \
		--volume="$(CURDIR)/.bash_history:/root/.bash_history" \
		--network host \
		--workdir /root/legged_rl_gym \
		$(IMAGE_NAME)

run_lpf:
	$(MAKE) run BRANCH=lpf

run_pure_amp:
	$(MAKE) run BRANCH=pure_amp

run_headless: checkout_branch
	touch .bash_history
	docker run -it --rm --name $(CONTAINER_NAME) \
		--gpus all \
		--privileged \
		--ipc=host \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		--env="NVIDIA_VISIBLE_DEVICES=all" \
		--env="NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics" \
		--env="MUJOCO_GL=egl" \
		--env="PYOPENGL_PLATFORM=egl" \
		--env="ROS_DOMAIN_ID=$(DOMAIN_ID)" \
		--env="LEGGED_RL_GYM_BRANCH=$(BRANCH)" \
		--volume="/dev/input:/dev/input:rw" \
		--volume="$(CURDIR)/$(REPO_DIR):/root/legged_rl_gym" \
		--volume="$(CURDIR)/.bash_history:/root/.bash_history" \
		--network host \
		--workdir /root/legged_rl_gym \
		$(IMAGE_NAME)

run_headless_lpf:
	$(MAKE) run_headless BRANCH=lpf

run_headless_pure_amp:
	$(MAKE) run_headless BRANCH=pure_amp

attach:
	docker exec -it $(CONTAINER_NAME) /bin/bash

shell: run

test: checkout_branch
	docker run --rm --gpus all \
		--volume="$(CURDIR)/$(REPO_DIR):/root/legged_rl_gym" \
		--workdir /root/legged_rl_gym \
		$(IMAGE_NAME) \
		bash -lc "python -c 'from isaacgym import gymapi, gymtorch; import torch; import legged_gym; import rsl_rl; import mujoco; import os; print(\"branch\", os.environ.get(\"LEGGED_RL_GYM_BRANCH\", \"unknown\")); print(\"isaacgym import ok\"); print(\"legged_gym\", legged_gym.LEGGED_GYM_ROOT_DIR); print(\"rsl_rl import ok\"); print(\"mujoco\", mujoco.__version__); print(\"torch\", torch.__version__, \"cuda\", torch.version.cuda); print(\"cuda_available\", torch.cuda.is_available()); print(\"device\", torch.cuda.get_device_name(0)); print(\"capability\", torch.cuda.get_device_capability(0)); x=torch.randn(1024,1024,device=\"cuda\"); print(float((x@x).mean()))'"

train_go1:
	docker exec -it $(CONTAINER_NAME) bash -lc "cd /root/legged_rl_gym && python legged_gym/scripts/train.py --task=go1_amp --headless"

train_go2:
	docker exec -it $(CONTAINER_NAME) bash -lc "cd /root/legged_rl_gym && python legged_gym/scripts/train.py --task=go2_amp --headless"

train_a1:
	docker exec -it $(CONTAINER_NAME) bash -lc "cd /root/legged_rl_gym && python legged_gym/scripts/train.py --task=a1_amp --headless"

clean:
	docker rm -f legged_rl_gym_lpf >/dev/null 2>&1 || true
	docker rm -f legged_rl_gym_pure_amp >/dev/null 2>&1 || true
