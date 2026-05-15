import numpy as np


def compute_lpf_alpha(cutoff_hz, dt):
    """Return the first-order low-pass filter alpha for a cutoff and timestep."""
    if cutoff_hz is None or cutoff_hz <= 0.0:
        return 1.0
    return float(1.0 - np.exp(-2.0 * np.pi * float(cutoff_hz) * float(dt)))


class ActionTargetLowPassFilter:
    """Low-pass filter for absolute joint-position targets."""

    def __init__(self, default_target, cutoff_hz, dt):
        self.default_target = np.asarray(default_target, dtype=np.float32)
        self.alpha = compute_lpf_alpha(cutoff_hz, dt)
        self.target = self.default_target.copy()

    def reset(self, target=None):
        if target is None:
            self.target = self.default_target.copy()
        else:
            self.target = np.asarray(target, dtype=np.float32).copy()
        return self.target.copy()

    def update(self, raw_target):
        raw_target = np.asarray(raw_target, dtype=np.float32)
        self.target = (self.target + self.alpha * (raw_target - self.target)).astype(np.float32)
        return self.target.copy()


def action_to_joint_target(action, config, target_filter):
    """Convert a normalized policy action to a filtered joint target."""
    clipped_action = np.clip(
        action[:12],
        -config.clip_actions,
        config.clip_actions,
    ).astype(np.float32)
    raw_target = clipped_action * config.action_scale + config.default_dof_pos
    raw_target = np.clip(raw_target, config.joint_limit_low, config.joint_limit_high)
    filtered_target = target_filter.update(raw_target)
    return clipped_action, filtered_target
