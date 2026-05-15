# Action Target Low-Pass Filter Implementation

This branch implements a first-order low-pass filter (LPF) on absolute
joint-position action targets in both training and deployment.

## Filter Model

The filter equation is:

```text
y[t] = y[t-1] + alpha * (x[t] - y[t-1])
alpha = 1 - exp(-2 * pi * cutoff_hz * dt)
```

Code meanings:

- `x[t]` is the unfiltered absolute joint-position target.
- `y[t]` is the filtered absolute joint-position target.
- `dt` is the action target update period.
- `cutoff_hz <= 0` or `null` sets `alpha = 1.0`, so the target passes through.

The branch default cutoff is `5.0 Hz`. For Go2 AMP training and deployment,
`dt = 0.005 * 6 = 0.03 s`, so `alpha ~= 0.610`.

## Training Path

Training code lives in `legged_gym/envs/base/legged_robot.py`.

The config key is `cfg.control.action_lpf_cutoff_hz`. The branch default is
`5.0` in `LeggedRobotCfg.control`, and `GO2AMPCfg.control` also sets `5.0`.

`LeggedRobot._init_buffers()` creates:

- `self.action_lpf_alpha`
- `self.action_targets`

`self.action_targets` is initialized to `default_dof_pos` for every env.

`LeggedRobot.step()` clips the policy action, updates filtered action targets
once, then enters the physics decimation loop. For Go2 AMP, that means one LPF
update per `0.03 s` policy step, followed by six Isaac Gym physics substeps at
`0.005 s`.

The target update is:

```text
raw_target = default_dof_pos + action_scale * clipped_policy_action
action_targets = action_targets + alpha * (raw_target - action_targets)
```

`LeggedRobot._compute_torques()` is still called once per physics substep. For
`control_type == "P"`, it computes:

```text
torque = p_gain * (action_targets - dof_pos) - d_gain * dof_vel
torque = clip(torque, -torque_limits, torque_limits)
```

Velocity (`"V"`) and torque (`"T"`) control paths still use the scaled policy
action directly. The LPF target buffer is not used by those paths.

`reset_idx()` resets `self.action_targets[env_ids]` to `default_dof_pos`.
Policy observations, `last_actions`, and the action-rate reward continue to use
the raw clipped policy action, not the filtered target.

## Deployment Path

Deployment code uses `deploy/action_lpf.py`.

The deployment config key is `action_lpf_cutoff_hz`; `deploy/config/go1.yaml`
and `deploy/config/go2.yaml` both set `5.0`. The keyboard loader and joystick
loader also default missing `action_lpf_cutoff_hz` to `5.0`.

Deployment alpha is computed from:

```text
dt = policy_decimation * sim_dt
policy_decimation = int(policy_dt / sim_dt)
```

For the current Go2 config, `policy_decimation = 6` and `sim_dt = 0.005`, so
deployment uses the same `0.03 s` target update period as training.

`action_to_joint_target()` performs this sequence:

```text
clipped_action = clip(policy_action, -clip_actions, clip_actions)
raw_target = default_dof_pos + action_scale * clipped_action
raw_target = clip(raw_target, joint_limit_low, joint_limit_high)
filtered_target = action_lpf(raw_target)
```

The filtered target is in training joint order `[FL, FR, RL, RR]`.

`sim2sim2real_keyboard.py` and `sim2sim2real_joystick.py` both use this helper
in MuJoCo sim mode and Unitree real mode. Sim mode sends the filtered target to
MuJoCo actuators. Real mode maps the filtered training-order target to SDK order
before writing Unitree `LowCmd` motor position fields.

Controller startup creates the LPF state at `default_dof_pos`. Standup and
stabilization phases send their own targets; the first policy-control target is
filtered from the default pose.
