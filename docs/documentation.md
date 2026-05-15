# legged_rl_gym Architecture And Development Notes

This document describes the checked-out `legged_rl_gym` repository as it exists
in this workspace. It is written for future coding agents that need to modify,
debug, train, export, or deploy policies without rediscovering the project
structure from scratch.

## Workspace Context

This repository is nested inside an execution wrapper at
`/home/roy/Documents/sshfs_mnt`. Read `../AMP_train_env.md` before doing
training or runtime work. The short version is:

- The Git repo is `legged_rl_gym/`, not the wrapper root.
- The current observed branch is `lpf`.
- The host Python environment is not the Isaac Gym training environment.
- Normal training uses Slurm plus the existing `legged_rl_gym.sif` Apptainer
  image.
- Do not use the Docker workflow on this server unless the user explicitly asks
  for image maintenance elsewhere.
- Slurm binds the host checkout to `/root/legged_rl_gym` inside the container,
  so jobs run the exact files in this host worktree.

Pre-existing local changes are present in this checkout, including modified
Go2 mocap files, docs, and `legged_gym/envs/go2/go2_amp_config.py`. Do not
discard or overwrite them unless the user asks.

## What This Repo Does

`legged_rl_gym` trains quadruped locomotion policies with Isaac Gym and a local
fork of `rsl_rl`. The main active task in this workspace is `go2_amp`, an
Adversarial Motion Prior policy for the Unitree Go2. The policy learns from:

- task rewards from the Isaac Gym environment, mainly velocity tracking and
  action smoothing penalties in the current Go2 AMP config;
- style rewards from an AMP discriminator trained against reference mocap;
- reference-state initialization from sampled mocap frames.

After training, `play.py` loads a checkpoint and exports the actor network as a
TorchScript `.pt` file. Deployment scripts then run the exported policy in
MuJoCo or on a Unitree robot through the Unitree SDK.

## Main Directory Map

Important files and directories:

```text
legged_gym/
  __init__.py                         root path constants
  envs/__init__.py                    task registration
  envs/base/base_config.py            recursive nested config instantiation
  envs/base/base_task.py              Isaac Gym task base class
  envs/base/legged_robot_config.py    base env and PPO config defaults
  envs/base/legged_robot.py           simulation, control, rewards, observations
  envs/base/observation_buffer.py     optional observation history buffer
  envs/go2/go2_amp_config.py          current Go2 AMP training config
  envs/go2/go2_config.py              Go2 non-AMP rough terrain config
  envs/go1/, envs/a1/                 analogous Go1/A1 configs
  scripts/train.py                    training entrypoint
  scripts/play.py                     checkpoint loading and policy export
  scripts/record_policy.py            viewer-based video recording
  utils/task_registry.py              task/env/runner factory
  utils/helpers.py                    CLI args, config conversion, checkpoint path

rsl_rl/rsl_rl/
  modules/actor_critic.py             feed-forward actor-critic MLP
  runners/amp_on_policy_runner.py     AMP rollout and training loop
  algorithms/amp_ppo.py               PPO plus discriminator optimization
  algorithms/amp_discriminator.py     discriminator and style reward
  datasets/motion_loader.py           reference motion loader/sampler
  storage/rollout_storage.py          PPO rollout storage
  storage/replay_buffer.py            AMP policy transition replay
  utils/utils.py                      normalizer and trajectory helpers

datasets/
  mocap_motions/                      original A1-style mocap JSON txt files
  mocap_go1/, mocap_go2/              retargeted motion files
  retarget_mocap.py                   A1 -> Go1/Go2 retargeting tool
  verify_mocap.py                     static motion validation
  replay_mocap.py                     Isaac Gym mocap playback without policy

deploy/
  config/go2.yaml                     Go2 deploy and sim2sim config
  config/go1.yaml                     Go1 deploy and sim2sim config
  sim2sim2real_keyboard.py            keyboard MuJoCo/real controller
  sim2sim2real_joystick.py            joystick MuJoCo/real controller
  real2sim.py                         read real robot state into sim ordering
  exported_policy/                    deployable TorchScript policies
  assets/go2/                         MuJoCo assets and scenes

resources/robots/go2/urdf/go2.urdf    Isaac Gym Go2 asset
logs/                                 training runs, checkpoints, exports
```

The bundled `isaacgym/` tree is a large dependency payload. Avoid scanning it
for project behavior unless debugging Isaac Gym itself.

## Runtime Commands

From the wrapper root:

```bash
cd /home/roy/Documents/sshfs_mnt
git -C legged_rl_gym status --short --branch
```

Submit the LPF-branch Go2 AMP workflow:

```bash
git -C legged_rl_gym checkout lpf
sbatch train_go2_amp_lpf.sbatch
```

Submit the pure AMP workflow:

```bash
sbatch train_go2_amp_pure.sbatch
```

Both Slurm scripts run:

```bash
apptainer exec --nv \
  --bind "${REPO}:/root/legged_rl_gym" \
  --pwd /root/legged_rl_gym \
  "${SIF}" \
  /opt/conda/envs/legged_rl/bin/python legged_gym/scripts/train.py \
    --task=go2_amp \
    --headless \
    --num_envs=8192 \
    --max_iterations=50000
```

The config default is `max_iterations = 500000`, but the Slurm launchers
override it to `50000`.

For manual container use, preserve the same bind mount and Python path. Host
Python is not a valid substitute for training unless the user has explicitly set
up Isaac Gym and the editable packages on the host.

## Task Registration And Entrypoints

Task names are registered in `legged_gym/envs/__init__.py`:

```python
task_registry.register("go2_amp", LeggedRobot, GO2AMPCfg(), GO2AMPCfgPPO())
```

Every registered task maps a name to:

- an environment class, currently `LeggedRobot` for all tasks;
- an environment config object, for example `GO2AMPCfg`;
- a training config object, for example `GO2AMPCfgPPO`.

`legged_gym/scripts/train.py` is intentionally thin:

```text
get_args()
task_registry.make_env(name=args.task)
task_registry.make_alg_runner(env=env, name=args.task)
runner.learn(...)
```

`TaskRegistry.make_env()`:

1. looks up the registered env class and config;
2. applies CLI overrides such as `--num_envs`;
3. seeds Python, NumPy, and Torch from the training config seed;
4. converts `cfg.sim` into Isaac Gym `SimParams`;
5. constructs the env class with the selected Isaac Gym devices.

`TaskRegistry.make_alg_runner()`:

1. looks up the training config;
2. applies CLI overrides such as `--resume`, `--load_run`, `--checkpoint`,
   `--max_iterations`, `--experiment_name`, and `--run_name`;
3. builds a log directory under `logs/<experiment_name>/<date>_<run_name>`;
4. evaluates `runner_class_name`, for Go2 AMP this is `AMPOnPolicyRunner`;
5. passes the environment and a dictified training config into the runner;
6. loads a checkpoint if resume is enabled.

The use of `eval()` means runner, algorithm, and policy class names must be
imported into the factory module's namespace. The current imports expose
`OnPolicyRunner`, `AMPOnPolicyRunner`, `PPO`, `AMPPPO`, `ActorCritic`, and
`ActorCriticRecurrent` through package `__init__.py` files.

## Config System

Configs are class trees, not YAML files. `BaseConfig.__init__()` recursively
instantiates nested classes, so this:

```python
class GO2AMPCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
```

becomes an object with `cfg.env.num_envs`.

`class_to_dict()` in `legged_gym/utils/helpers.py` recursively converts those
config objects into dictionaries for the runner/algorithm layer.

Key base defaults live in `legged_gym/envs/base/legged_robot_config.py`:

- `env`: env count, observation sizes, action count, episode length,
  reference-state initialization toggle.
- `terrain`: plane, heightfield, or trimesh terrain generation settings.
- `commands`: velocity command ranges and resampling timing.
- `init_state`: root pose and default joint angles.
- `control`: control type, PD gains, action scale, and control decimation.
- `asset`: URDF/MJCF path and body-name filters for feet, collisions, resets.
- `domain_rand`: friction, mass, pushes, and gain randomization.
- `rewards`: reward names and scales.
- `normalization`: observation scales and clipping.
- `noise`: observation noise scales.
- `sim`: Isaac Gym PhysX settings.

## Current Go2 AMP Config

The active Go2 AMP config is `legged_gym/envs/go2/go2_amp_config.py`.

Important current values:

```text
MOTION_FILES = glob.glob("datasets/mocap_go2/*")
env.num_envs = 4096, overridden to 8192 by Slurm
env.num_observations = 45
env.num_privileged_obs = 48
env.num_actions = 12 inherited from base config
env.reference_state_initialization = True
env.reference_state_initialization_prob = 0.85
asset.file = resources/robots/go2/urdf/go2.urdf
terrain.mesh_type = plane
terrain.measure_heights = False
control.control_type = P
control.stiffness = {"joint": 25}
control.damping = {"joint": 0.5}
control.action_scale = 0.25
control.action_lpf_cutoff_hz = 5.0
control.decimation = 6
sim.dt = 0.005 inherited from base config
policy dt = sim.dt * decimation = 0.03 s
action LPF alpha ~= 0.610
runner_class_name = AMPOnPolicyRunner
algorithm_class_name = AMPPPO
policy_class_name = ActorCritic
experiment_name = go2_amp_example
amp_reward_coef = 2.0
amp_num_preload_transitions = 200000
amp_task_reward_lerp = 0.3
amp_discr_hidden_dims = [1024, 512]
min_normalized_std = [0.05, 0.02, 0.05] * 4
```

The Go2 AMP task reward scales currently leave most base rewards at zero and
use:

- `tracking_lin_vel = 1.5 / 0.03`
- `tracking_ang_vel = 0.5 / 0.03`
- `action_rate = -0.01`

Reward scales are later multiplied by `dt` in `_prepare_reward_function()`, so
the explicit division by `0.03` makes the effective tracking weights close to
the intended constants after dt scaling.

The current Go2 AMP command config uses:

```text
heading_command = False
smooth_command_changes = True
smooth_command_alpha = 0.99
lin_vel_x = [-1.0, 2.2]
lin_vel_y = [-0.3, 0.3]
ang_vel_yaw = [-1.57, 1.57]
```

`smooth_command_changes` is implemented in `LeggedRobot`: resampling updates
`command_targets`, and each env step applies an exponential moving average to
`self.commands`.

## LeggedRobot Lifecycle

`LeggedRobot` inherits `BaseTask`. Construction does this:

1. `_parse_cfg()` computes derived values:
   - `self.dt = cfg.control.decimation * sim_params.dt`
   - reward scale dict
   - command range dict
   - max episode length in policy steps
   - push interval in policy steps
2. `BaseTask.__init__()` creates the Isaac Gym object, devices, tensors, sim,
   and viewer.
3. `LeggedRobot.create_sim()` creates the PhysX sim, terrain, and envs.
4. `_create_envs()` loads the robot asset, creates all env instances, applies
   rigid shape/body/DOF processing, and resolves feet/contact body indices.
5. `_init_buffers()` wraps Isaac Gym root, DOF, and contact tensors and creates
   Torch buffers for actions, commands, rewards, observations, velocities, PD
   gains, defaults, and optional command targets.
6. `_prepare_reward_function()` removes zero reward scales, multiplies nonzero
   scales by `dt`, and stores function pointers named `_reward_<name>`.
7. If reference-state initialization is enabled, an `AMPLoader` is created for
   reset-time sampling.

### Step Loop

`LeggedRobot.step(actions)`:

1. clips incoming actions by `cfg.normalization.clip_actions`;
2. repeats `cfg.control.decimation` physics substeps;
3. on each substep computes torques with `_compute_torques()`;
4. writes torques into Isaac Gym and simulates;
5. calls `post_physics_step()`;
6. clips observations;
7. optionally updates observation history;
8. returns:

```text
policy_obs,
privileged_obs,
rew_buf,
reset_buf,
extras,
reset_env_ids,
terminal_amp_states
```

The 7-value return is important. Some older code, including
`legged_gym/tests/test_env.py`, still unpacks the pre-AMP 5-value step return.

### Post Physics

`post_physics_step()`:

1. refreshes actor root states and contact forces;
2. increments episode counters;
3. computes base linear/angular velocities in the base frame;
4. computes projected gravity;
5. calls `_post_physics_step_callback()`;
6. checks termination;
7. computes task rewards;
8. snapshots terminal AMP states for envs about to reset;
9. resets done envs;
10. recomputes observations;
11. updates `last_actions`, `last_dof_vel`, and `last_root_vel`.

`_post_physics_step_callback()` resamples commands every
`cfg.commands.resampling_time / self.dt` policy steps. With command smoothing
enabled, it moves `self.commands` toward `self.command_targets` every step:

```text
commands = alpha * commands + (1 - alpha) * command_targets
```

It also computes heading-derived yaw commands when `heading_command` is true,
updates measured terrain heights when enabled, and applies random pushes at the
configured interval.

### Control And Torques

For `control_type == "P"`, the clipped policy action is converted to an
absolute joint target once per policy step, before the physics decimation loop:

```text
raw_target = default_dof_pos + action_scale * actions
action_targets = action_targets + action_lpf_alpha * (raw_target - action_targets)
torque = p_gain * (action_targets - dof_pos) - d_gain * dof_vel
torque = clip(torque, -torque_limits, torque_limits)
```

With Go2 AMP defaults, action zero means the configured standing pose, and each
action unit is `0.25 rad` around that pose. The branch default
`action_lpf_cutoff_hz = 5.0`; with `self.dt = 0.03 s`, `action_lpf_alpha` is
about `0.610`.

`_compute_torques()` runs once per physics substep and consumes the held
`action_targets` value. Velocity and torque control modes still use the scaled
policy action directly.

Domain-randomized gains replace `p_gains` and `d_gains` with per-env randomized
versions when `cfg.domain_rand.randomize_gains` is true.

### Resets

`reset_idx(env_ids)` handles terrain/command curriculum, robot state reset,
command resampling, gain randomization, buffer clearing, and episode logging.
It resets `action_targets[env_ids]` to `default_dof_pos`.

If `cfg.env.reference_state_initialization` is true, it samples full mocap
frames with `self.amp_loader.get_full_frame_batch(len(env_ids))` and resets:

- DOF positions from `AMPLoader.get_joint_pose_batch(frames)`;
- DOF velocities from `AMPLoader.get_joint_vel_batch(frames)`;
- root position from mocap root position plus env origin;
- root orientation from mocap root quaternion;
- root linear/angular velocities rotated from mocap local frame into world.

The config exposes `reference_state_initialization_prob`, but the current
`reset_idx()` implementation always uses AMP reset frames when
`reference_state_initialization` is true. There is no probability branch in the
current code.

### Observations

`compute_observations()` first builds `privileged_obs_buf`:

```text
base_lin_vel * lin_vel_scale                 3
base_ang_vel * ang_vel_scale                 3
projected_gravity                            3
commands[:, :3] * commands_scale             3
(dof_pos - default_dof_pos) * dof_pos_scale 12
dof_vel * dof_vel_scale                     12
actions                                     12
------------------------------------------------
total                                       48
```

If terrain height measurement is enabled, measured heights are appended. Go2 AMP
has it disabled.

Policy observations are derived from privileged observations:

- if `num_observations == num_privileged_obs - 3`, drop only base linear
  velocity; this is Go2 AMP's 45-dim policy observation;
- if `num_observations == num_privileged_obs - 6`, drop base linear velocity
  and commands;
- otherwise clone privileged observations.

For Go2 AMP, the exported/deploy policy expects this 45-dim observation:

```text
base_ang_vel * 0.25                 3
projected_gravity                   3
commands * [2.0, 2.0, 0.25]         3
dof_pos - default_dof_pos          12
dof_vel * 0.05                     12
last action                        12
```

This matches the observation builders in the deployment scripts.

### AMP Observations

`get_amp_observations()` returns:

```text
dof_pos                                      12
foot_positions_in_base_frame(dof_pos)       12
base_lin_vel                                 3
base_ang_vel                                 3
dof_vel                                     12
root_z                                      1
------------------------------------------------
total                                       43
```

The AMP discriminator sees current and next AMP observations concatenated, so
the current discriminator input dimension is `86`.

Important caveat: `foot_positions_in_base_frame()` uses hardcoded A1-like
geometry constants in `LeggedRobot` (`HIP_OFFSETS`, `l_up = 0.2`,
`l_low = 0.2`, `l_hip = 0.08505`). The Go2 mocap retargeting script uses Go2
dimensions, but the training env's AMP foot reconstruction currently does not
read robot-specific geometry from the Go2 URDF or config.

## Rewards

Rewards are selected dynamically from nonzero `cfg.rewards.scales` values. To
add a reward:

1. add a scale under `cfg.rewards.scales`;
2. implement `_reward_<name>(self)` in `LeggedRobot`;
3. ensure the reward returns a tensor shaped `[num_envs]`.

Common existing reward functions include:

- `_reward_tracking_lin_vel`: `exp(-||command_xy - base_lin_vel_xy||^2 /
  tracking_sigma)`.
- `_reward_tracking_ang_vel`: `exp(-(command_yaw - base_ang_vel_yaw)^2 /
  tracking_sigma)`.
- `_reward_action_rate`: sum squared action change.
- `_reward_collision`: counts selected body contacts.
- `_reward_dof_pos_limits`: penalizes positions outside soft limits.
- `_reward_feet_air_time`: rewards long steps on first contact, active only
  when command xy norm is above `0.1`.

If `only_positive_rewards` is true, total reward is clipped to be nonnegative
before the termination reward is added.

For AMP training, `LeggedRobot.compute_reward()` produces task rewards, but
`AMPOnPolicyRunner.learn()` replaces the reward used by PPO with
`discriminator.predict_amp_reward(...)`. When `amp_task_reward_lerp > 0`, the
discriminator style reward is linearly mixed with the task reward.

## Motion Data And AMPLoader

Motion files are JSON stored with `.txt` extensions. Each file has metadata and
a `Frames` array. A raw frame has 61 values in PyBullet leg order
`[FR, FL, RR, RL]`, with each leg ordered `[hip, thigh, calf]`:

```text
root_pos          0:3    3
root_rot          3:7    4  xyzw quaternion
joint_pos         7:19  12
toe_pos_local    19:31  12
linear_vel       31:34   3
angular_vel      34:37   3
joint_vel        37:49  12
toe_vel_local    49:61  12
```

`rsl_rl/rsl_rl/datasets/motion_loader.py` loads these files through
`AMPLoader`. At load time it:

1. reorders leg data from PyBullet `[FR, FL, RR, RL]` to Isaac/training
   `[FL, FR, RL, RR]`;
2. normalizes and standardizes root quaternions;
3. stores a compact AMP trajectory without root pose:
   `joint_pos + toe_pos + linear_vel + angular_vel + joint_vel` (42 dims);
4. stores full frames through `JOINT_VEL_END_IDX` for reset and preloading;
5. records trajectory weights, lengths, frame durations, and frame counts.

`AMPLoader.observation_dim` returns compact trajectory width plus root z, so the
current AMP observation dimension is `42 + 1 = 43`.

When `preload_transitions=True`, the loader samples many `(s, s_next)` pairs
once at startup. The Go2 AMP runner currently requests `200000` preloaded
transitions. The preloaded expert transitions append root z to both states.

`AMPLoader.feed_forward_generator()` yields expert minibatches for the
discriminator. `ReplayBuffer.feed_forward_generator()` yields policy
transitions collected from the running env. The discriminator trains on both.

### Retargeting And Verification

`datasets/retarget_mocap.py` retargets from A1-style mocap to Go1 or Go2:

- scales root z by total limb-length ratio;
- leaves joint angles unchanged;
- recomputes toe positions through target-robot forward kinematics;
- scales linear velocity and toe velocity;
- leaves angular velocity and joint velocity unchanged.

Useful commands from the repo root, inside a valid Python environment:

```bash
python datasets/retarget_mocap.py --src a1 --tgt go2
python datasets/verify_mocap.py --dir datasets/mocap_go2 --robot go2 --plot
python datasets/replay_mocap.py --task go2_amp --speed 0.5
```

`datasets/replay_mocap.py` creates a one-env Isaac Gym task, disables
randomization, and writes mocap root/DOF state tensors directly. It is useful
for visually validating motion data independently from policy learning.

## AMP Training Stack

### Runner

`rsl_rl/rsl_rl/runners/amp_on_policy_runner.py` owns the high-level training
loop.

During initialization it:

1. determines actor observation size and critic observation size;
2. builds `ActorCritic`;
3. builds an expert `AMPLoader`;
4. builds a running `Normalizer` for AMP observations;
5. builds `AMPDiscriminator(input_dim=86, hidden_dims=[1024, 512])`;
6. builds `AMPPPO`;
7. initializes PPO rollout storage;
8. resets the env.

During `learn()` it:

1. gets policy observations, privileged critic observations, and AMP obs;
2. collects `num_steps_per_env` rollout steps, default `24`;
3. calls `AMPPPO.act()` to sample actions and record old policy statistics;
4. calls `env.step(actions)`;
5. substitutes terminal AMP states for envs that reset so discriminator
   transitions use the true terminal next state;
6. computes AMP reward with `discriminator.predict_amp_reward(...)`;
7. stores PPO and AMP transitions;
8. computes returns;
9. calls `AMPPPO.update()`;
10. logs TensorBoard scalars and stdout metrics;
11. saves checkpoints every `save_interval`, default `50`.

Checkpoints contain:

```text
model_state_dict
optimizer_state_dict
discriminator_state_dict
amp_normalizer
iter
infos
```

### ActorCritic

`rsl_rl/rsl_rl/modules/actor_critic.py` defines the feed-forward actor-critic:

- actor MLP maps policy observation to 12 action means;
- critic MLP maps privileged observations to scalar value;
- action distribution is a diagonal Gaussian with learned std unless
  `fixed_std=True`;
- `act()` samples actions for training;
- `act_inference()` returns deterministic action means for export/deploy.

Go2 AMP inherits the base hidden sizes:

```text
actor_hidden_dims = [512, 256, 128]
critic_hidden_dims = [512, 256, 128]
activation = elu
init_noise_std = 1.0
```

### AMPPPO

`rsl_rl/rsl_rl/algorithms/amp_ppo.py` combines PPO and discriminator training.

The PPO part:

- stores observations, critic observations, actions, values, log probs, action
  means/stds, rewards, and dones in `RolloutStorage`;
- computes GAE returns;
- uses the clipped PPO surrogate;
- optionally adapts learning rate based on KL to `desired_kl`;
- uses clipped value loss when enabled;
- clamps learned action std to `min_normalized_std` converted into joint-space
  units from DOF limits.

The AMP part:

- stores policy AMP transitions in `ReplayBuffer`;
- samples expert transitions from `AMPLoader`;
- normalizes policy and expert AMP states with `Normalizer`;
- trains the discriminator with MSE targets:
  - expert -> `+1`
  - policy -> `-1`
- adds a gradient penalty on expert transitions;
- updates the normalizer from both policy and expert states.

`AMPDiscriminator.predict_amp_reward()` computes:

```text
style_reward = amp_reward_coef * clamp(1 - 0.25 * (D(s, s_next) - 1)^2, min=0)
```

If `task_reward_lerp > 0`, final reward is:

```text
(1 - task_reward_lerp) * style_reward + task_reward_lerp * task_reward
```

For Go2 AMP, `task_reward_lerp = 0.3`.

## Policy Export And Deployment

### Export

`legged_gym/scripts/play.py` is the standard export path. It:

1. loads task configs;
2. shrinks the env to at most 50 robots;
3. disables terrain curriculum, noise, friction randomization, pushes, gain
   randomization, and mass randomization;
4. sets `amp_num_preload_transitions = 1` for faster startup;
5. creates the env and loads the latest checkpoint by setting
   `train_cfg.runner.resume = True`;
6. exports the actor with `export_policy_as_jit()`.

The export path used by `play.py` is:

```text
logs/<experiment_name>/exported/policies/policy_1.pt
```

The Slurm scripts then copy that file to:

```text
deploy/exported_policy/go2/policy.pt
```

`export_policy_as_jit()` saves only the actor network for feed-forward
policies. It does not include the critic, discriminator, normalizer, or Isaac
Gym dependencies.

### Deployment Config

`deploy/config/go2.yaml` must stay aligned with training:

- `default_dof_pos` must match the training default joint angles in training
  order `[FL, FR, RL, RR]`.
- `action_scale` must match `cfg.control.action_scale`.
- observation scales must match `cfg.normalization.obs_scales`.
- `sim_dt` is `0.005`.
- `control_decimation` is `6`, giving a `0.03 s` policy/control update.
- `policy_hz` is `33`, matching roughly `1 / 0.03`.
- `action_lpf_cutoff_hz` is `5.0`, giving `alpha ~= 0.610` at the actual
  decimated target update period.
- joint order maps convert between training order and Unitree SDK order.

The Go2 deployment observation builder uses the 45-dim policy observation:

```text
base_ang_vel,
projected_gravity,
commands,
dof_pos - default_dof_pos,
dof_vel,
last_action
```

It does not include base linear velocity, matching Go2 AMP training.

### Sim2Sim

`deploy/sim2sim2real_keyboard.py --mode sim --config config/go2.yaml` runs the
TorchScript policy in MuJoCo. It:

1. loads the configured MuJoCo scene;
2. creates joint and actuator name lists from `leg_order` and `joint_suffixes`;
3. loads the TorchScript policy;
4. stands the robot up by interpolating to default joint angles;
5. stabilizes briefly;
6. every policy interval, builds the 45-dim observation, runs the policy,
   scales actions to joint targets, clips targets by joint limits, filters the
   targets in training joint order, and sends targets to MuJoCo;
7. renders at a decimated rate for performance.

The configured Go2 scene is currently `deploy/assets/go2/scene_pos.xml`, which
uses MuJoCo position servos. The code can also handle torque actuators by
computing PD torques in Python.

### Sim2Real

`deploy/sim2sim2real_keyboard.py --mode real --config config/go2.yaml` uses the
Unitree SDK. It:

1. opens UDP low-level control;
2. waits for valid robot state;
3. maps SDK joint states to training order;
4. builds the same 45-dim observation as sim2sim;
5. runs the policy at the configured decimated rate;
6. scales, clips, and filters training-order targets;
7. maps training-order targets back to SDK order;
8. sends position commands with configured walking gains.

Safety caveat: the current `Sim2RealController._control_step()` contains an
auto-test behavior that injects a forward command when all operator commands
are zero and `sim_time < 15.0`. Treat that as bring-up/debug code, not a
neutral deployment behavior.

`deploy/sim2sim2real_joystick.py` follows the same policy/control structure but
uses a gamepad thread and smoothed joystick velocities.

## Current LPF Branch State

The `lpf` branch implements an action-target low-pass filter in training and
deployment. The detailed implementation note is
`docs/low_pass_filter_implementation.md`.

Training facts:

- `LeggedRobotCfg.control.action_lpf_cutoff_hz = 5.0`.
- `GO2AMPCfg.control.action_lpf_cutoff_hz = 5.0`.
- `LeggedRobot.step()` updates filtered position targets once per policy step,
  before the physics decimation loop.
- `LeggedRobot._compute_torques()` runs once per physics substep and uses the
  held `self.action_targets` value for P control.
- `reset_idx()` resets selected env filter state to `default_dof_pos`.
- Policy observations, `last_actions`, and action-rate reward use raw clipped
  policy actions.

Deployment facts:

- `deploy/action_lpf.py` contains the NumPy LPF implementation.
- `deploy/config/go1.yaml` and `deploy/config/go2.yaml` set
  `action_lpf_cutoff_hz: 5.0`.
- Keyboard and joystick deployment controllers use the filter in both sim and
  real modes.
- Deployment computes LPF `dt` from `policy_decimation * sim_dt`, which is
  `0.03 s` for the current Go2 config.
- The filtered target is kept in training joint order until the real-mode SDK
  mapping step.

## How To Add Or Change Things

### Add A New Task

1. Create or extend a config file under `legged_gym/envs/<robot>/`.
2. Subclass `LeggedRobotCfg` and `LeggedRobotCfgPPO`.
3. Set observation sizes, asset path, default angles, control gains,
   command ranges, and reward scales.
4. Register the task in `legged_gym/envs/__init__.py`:

```python
task_registry.register("my_task", LeggedRobot, MyCfg(), MyCfgPPO())
```

5. Ensure `runner_class_name`, `algorithm_class_name`, and `policy_class_name`
   are importable where `eval()` is used.

### Change Policy Observations

1. Change `cfg.env.num_observations` and, if needed, `num_privileged_obs`.
2. Update `LeggedRobot.compute_observations()`.
3. Update deployment observation builders if exported policies are used outside
   Isaac Gym.
4. Any checkpoint trained with the old observation size will not load into an
   actor with the new input size.

### Change AMP Observations

1. Update `LeggedRobot.get_amp_observations()`.
2. Update `AMPLoader.feed_forward_generator()` and `AMPLoader.observation_dim`
   so expert states have the same layout and dimension.
3. Update reset logic if the change depends on full mocap frames.
4. Existing discriminator checkpoints will not be compatible if the input size
   changes.

### Add A Reward

1. Add `cfg.rewards.scales.<name>`.
2. Implement `LeggedRobot._reward_<name>()`.
3. Remember that nonzero scales are multiplied by `self.dt`.
4. For AMP tasks, decide whether task reward should affect learning through
   `amp_task_reward_lerp`.

### Change Robot Geometry

1. Update the Isaac Gym asset under `resources/robots/...`.
2. Update task asset config path and default joint angles.
3. Update mocap retargeting parameters in `datasets/retarget_mocap.py`.
4. Check `LeggedRobot.foot_position_in_hip_frame()` and
   `HIP_OFFSETS`; these are hardcoded and currently not robot-config driven.
5. Update deploy MuJoCo assets and deploy YAML defaults/joint maps.
6. Verify with `verify_mocap.py`, `replay_mocap.py`, `play.py`, then MuJoCo
   sim2sim before hardware.

### Change Deployment Behavior

Keep these pieces consistent:

- training observation layout and deployment observation builder;
- training `default_joint_angles` and deploy `default_dof_pos`;
- training `action_scale` and deploy `action_scale`;
- training `action_lpf_cutoff_hz` and deploy `action_lpf_cutoff_hz`;
- training joint order `[FL, FR, RL, RR]` and SDK/MuJoCo mapping arrays;
- training policy dt (`sim.dt * decimation`) and deploy policy interval;
- target clipping and safety limits.

## Known Gotchas

- `README.md` contains useful high-level context but is partly stale relative
  to the current source tree. Trust source code over README examples when they
  disagree.
- The active env class for `go2_amp` is `LeggedRobot`; there is no separate
  `go2_amp.py` env class in this tree.
- `LeggedRobot.step()` returns 7 values in AMP mode. Older scripts/tests that
  expect 5 values need updating.
- `reference_state_initialization_prob` is configured but not used by the
  current reset implementation.
- AMP discriminator observation size is 43 per state and 86 per transition in
  the current code, because root z is appended to the 42-dim mocap state.
- Training and deployment assume training joint order `[FL, FR, RL, RR]`.
  Raw mocap is `[FR, FL, RR, RL]`, and Unitree SDK order is `[FR, FL, RR, RL]`.
- The Go2 AMP env's AMP foot-position calculation uses hardcoded A1 geometry.
- `play.py` exports to `logs/<experiment>/exported/policies/policy_1.pt`;
  Slurm copies that to `deploy/exported_policy/go2/policy.pt`.
- Hardware control code contains debug prints and an auto-forward test command;
  review and remove debug behaviors before serious deployment.

## Development And Verification Commands

Use `rg` for repository search:

```bash
rg "go2_amp" legged_gym rsl_rl deploy datasets
rg "get_amp_observations|observation_dim|feed_forward_generator" legged_gym rsl_rl
```

Compile-check edited Python files on the host when imports do not require
Isaac Gym:

```bash
python -m py_compile path/to/file.py
```

For Isaac Gym code, run inside the Apptainer image or Slurm environment. Small
smoke tests should use reduced env counts:

```bash
apptainer exec --nv \
  --bind /home/roy/Documents/sshfs_mnt/legged_rl_gym:/root/legged_rl_gym \
  --pwd /root/legged_rl_gym \
  /home/roy/Documents/sshfs_mnt/legged_rl_gym.sif \
  /opt/conda/envs/legged_rl/bin/python legged_gym/scripts/train.py \
    --task=go2_amp --headless --num_envs=32 --max_iterations=1
```

Export latest policy:

```bash
apptainer exec --nv \
  --bind /home/roy/Documents/sshfs_mnt/legged_rl_gym:/root/legged_rl_gym \
  --pwd /root/legged_rl_gym \
  /home/roy/Documents/sshfs_mnt/legged_rl_gym.sif \
  /opt/conda/envs/legged_rl/bin/python legged_gym/scripts/play.py \
    --task=go2_amp --headless --num_envs=1
```

Run MuJoCo sim2sim from `deploy/` when dependencies are available:

```bash
cd deploy
python sim2sim2real_keyboard.py --mode sim --config config/go2.yaml
```

For static motion checks:

```bash
python datasets/verify_mocap.py --dir datasets/mocap_go2 --robot go2 --plot
```

For job monitoring from the wrapper root:

```bash
squeue -u "$USER"
tail -f logs/train_go2_amp_lpf-<jobid>.out
tail -f slurm-pure-amp-<jobid>.out
```

## Practical Agent Checklist

Before editing:

1. Check `git -C /home/roy/Documents/sshfs_mnt/legged_rl_gym status --short`.
2. Read `../AMP_train_env.md` if the task touches training or runtime.
3. Identify whether the change affects training only, deployment only, or both.
4. Preserve existing local modifications unless the user explicitly asks for a
   revert.

Before finishing:

1. Run the narrowest viable verification command.
2. If tests cannot run because Isaac Gym/container/GPU is unavailable, state
   that clearly.
3. For observation, action, default-pose, or joint-order changes, audit the
   deploy YAML and deployment observation builders.
4. For AMP observation changes, audit both env and `AMPLoader`.
5. For hardware-facing changes, verify in MuJoCo sim2sim first.
