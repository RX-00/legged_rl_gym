#!/usr/bin/env python3

import argparse
import os
import sys
import time
import threading

import numpy as np
import torch
import yaml

from joystick import RemoteController, apply_deadzone
from action_lpf import ActionTargetLowPassFilter, action_to_joint_target

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient


POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0


# =============================================================================
# Config / observation helpers
# =============================================================================

def load_config(config_path):
    config_path = os.path.abspath(config_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    class Config:
        def __init__(self, d):
            for key, value in d.items():
                if isinstance(value, str):
                    value = value.replace(
                        "{LEGGED_GYM_ROOT_DIR}",
                        os.environ.get("LEGGED_GYM_ROOT_DIR", ""),
                    )
                    if value.startswith("./") or value.startswith("../"):
                        value = os.path.abspath(os.path.join(script_dir, value))

                if isinstance(value, dict):
                    setattr(self, key, value)
                elif isinstance(value, list):
                    if key in ("sdk_joint_order", "leg_order", "joint_suffixes"):
                        setattr(self, key, value)
                    elif key in ("train_to_sdk_map", "sdk_to_train_map"):
                        setattr(self, key, np.array(value, dtype=np.int64))
                    else:
                        try:
                            setattr(self, key, np.array(value, dtype=np.float32))
                        except Exception:
                            setattr(self, key, value)
                else:
                    setattr(self, key, value)

            if hasattr(self, "policy_hz"):
                self.policy_dt = 1.0 / float(self.policy_hz)

            if not hasattr(self, "action_lpf_cutoff_hz"):
                self.action_lpf_cutoff_hz = 5.0

            if hasattr(self, "vx_range"):
                self.vx_range = tuple(float(x) for x in self.vx_range)
            if hasattr(self, "vy_range"):
                self.vy_range = tuple(float(x) for x in self.vy_range)
            if hasattr(self, "vyaw_range"):
                self.vyaw_range = tuple(float(x) for x in self.vyaw_range)

    return Config(cfg)


def quat_from_euler_xyz(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return np.array([x, y, z, w], dtype=np.float32)


def quat_rotate_inverse(q, v):
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a - b + c


def compute_projected_gravity(quat_xyzw):
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return quat_rotate_inverse(quat_xyzw, gravity_world).astype(np.float32)


def build_obs(
    base_ang_vel,
    projected_gravity,
    commands,
    dof_pos,
    dof_vel,
    last_action,
    config,
):
    obs = []

    obs.extend(list(base_ang_vel * config.obs_scales["ang_vel"]))
    obs.extend(list(projected_gravity))

    commands_scaled = commands * config.obs_scales["commands"]
    obs.extend(list(commands_scaled))

    pos_delta = (dof_pos - config.default_dof_pos) * config.obs_scales["dof_pos"]
    obs.extend(list(pos_delta))

    obs.extend(list(dof_vel * config.obs_scales["dof_vel"]))
    obs.extend(list(last_action))

    return np.array(obs, dtype=np.float32)


def normalize_obs(obs, clip_observations=None):
    if clip_observations is None:
        return obs
    try:
        clip = float(clip_observations)
    except Exception:
        return obs
    return np.clip(obs, -clip, clip).astype(np.float32)


def resolve_policy_path(config, model_arg):
    if model_arg:
        # If config.policy_path is a file, put model next to it.
        if os.path.splitext(config.policy_path)[1]:
            return os.path.join(os.path.dirname(config.policy_path), model_arg)
        return os.path.join(config.policy_path, model_arg)

    return config.policy_path


# =============================================================================
# Gamepad controller
# =============================================================================

class GamepadController:
    """
    Logitech F710-style gamepad interface using deploy/joystick.py.

    Left stick:
      up/down    -> vx
      left/right -> vy

    Right stick:
      left/right -> yaw

    D-pad:
      up/down    -> incremental vx setpoint

    Start:
      exit
    """

    def __init__(
        self,
        vx_range=(0.0, 1.2),
        vy_range=(-0.3, 0.3),
        vyaw_range=(-1.57, 1.57),
    ):
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0

        self.vx_range = vx_range
        self.vy_range = vy_range
        self.vyaw_range = vyaw_range

        self.lock = threading.Lock()
        self.running = True
        self.exit_requested = False
        self.thread = None

        try:
            self.gamepad = RemoteController()
            self.gamepad.start()
            print("Gamepad initialized successfully")
        except Exception as exc:
            print(f"Failed to initialize gamepad: {exc}")
            self.gamepad = None

        self.deadzone = 0.05

        self.vx_increment = 0.1
        self.dpad_last_state = {"up": False, "down": False}

    def get_velocity(self):
        with self.lock:
            return self.vx, self.vy, self.vyaw

    def set_velocity(self, vx, vy, vyaw):
        with self.lock:
            self.vx = float(np.clip(vx, self.vx_range[0], self.vx_range[1]))
            self.vy = float(np.clip(vy, self.vy_range[0], self.vy_range[1]))
            self.vyaw = float(np.clip(vyaw, self.vyaw_range[0], self.vyaw_range[1]))

    def gamepad_thread(self):
        if self.gamepad is None:
            print("Gamepad unavailable; commands remain zero.")
            return

        update_interval = 1.0 / 33.0

        while self.running:
            try:
                loop_start = time.time()

                left_x, left_y = self.gamepad.get_left_stick(normalize=True)
                right_x, _ = self.gamepad.get_right_stick(normalize=True)

                left_x = apply_deadzone(left_x, self.deadzone)
                left_y = apply_deadzone(left_y, self.deadzone)
                right_x = apply_deadzone(right_x, self.deadzone)

                with self.gamepad.lock:
                    dpad_y = self.gamepad.axes[7] if len(self.gamepad.axes) > 7 else 0

                dpad_up_pressed = dpad_y < -16000
                dpad_down_pressed = dpad_y > 16000

                vx, vy, vyaw = self.get_velocity()

                if dpad_up_pressed and not self.dpad_last_state["up"]:
                    vx = min(vx + self.vx_increment, self.vx_range[1])
                    print(f"\n[D-pad UP] vx step: {vx:.2f} m/s")

                if dpad_down_pressed and not self.dpad_last_state["down"]:
                    vx = max(vx - self.vx_increment, 0.0)
                    print(f"\n[D-pad DOWN] vx step: {vx:.2f} m/s")

                self.dpad_last_state["up"] = dpad_up_pressed
                self.dpad_last_state["down"] = dpad_down_pressed

                # Left stick Y: push up is usually negative.
                if abs(left_y) > 0.1:
                    if left_y <= 0.0:
                        vx = (-left_y) * self.vx_range[1]
                    else:
                        vx = 0.0

                # Left stick X maps to lateral velocity.
                vy = -left_x * self.vy_range[1]

                # Right stick X maps to yaw velocity.
                vyaw = -right_x * self.vyaw_range[1]

                self.set_velocity(vx, vy, vyaw)

                if self.gamepad.is_button_pressed(self.gamepad.BTN_START):
                    print("\nStart button pressed; exiting.")
                    self.exit_requested = True
                    break

                elapsed = time.time() - loop_start
                sleep_time = max(0.0, update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as exc:
                print(f"\nGamepad error: {exc}")
                time.sleep(0.1)

    def start(self):
        self.thread = threading.Thread(target=self.gamepad_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.gamepad:
            self.gamepad.stop()
        if self.thread:
            self.thread.join(timeout=1.0)


# =============================================================================
# SDK2 Go2 real controller
# =============================================================================

class Go2SDK2JoystickController:
    def __init__(self, config, policy_path, network):
        self.config = config
        self.network = network

        self.joint_order = getattr(
            config,
            "sdk_joint_order",
            [
                "FR_0", "FR_1", "FR_2",
                "FL_0", "FL_1", "FL_2",
                "RR_0", "RR_1", "RR_2",
                "RL_0", "RL_1", "RL_2",
            ],
        )
        self.index = {name: i for i, name in enumerate(self.joint_order)}

        print(f"Initializing SDK2/DDS on network interface: {network}")
        ChannelFactoryInitialize(0, network)

        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = None
        self.low_state_lock = threading.Lock()
        self.crc = CRC()

        self._init_low_cmd()

        self.lowcmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_pub.Init()

        self.lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_sub.Init(self._lowstate_callback, 10)

        self.release_motion_mode()

        print(f"Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()

        self.last_action = np.zeros(12, dtype=np.float32)
        self.qDes_train = np.array(config.default_dof_pos, dtype=np.float32)

        self.policy_decimation = int(config.policy_dt / config.sim_dt)
        self.policy_counter = 0

        self.action_filter = ActionTargetLowPassFilter(
            config.default_dof_pos,
            config.action_lpf_cutoff_hz,
            self.policy_decimation * config.sim_dt,
        )
        self.qDes_train = self.action_filter.reset()

        self.current_qDes_sdk = None
        self.current_kp = config.kp_walk
        self.current_kd = config.kd_walk

        print("Go2 SDK2 joystick controller initialized")

    def _init_low_cmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0

        for i in range(20):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = POS_STOP_F
            self.low_cmd.motor_cmd[i].dq = VEL_STOP_F
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0

    def _lowstate_callback(self, msg):
        with self.low_state_lock:
            self.low_state = msg

    def _latest_state(self):
        with self.low_state_lock:
            return self.low_state

    def release_motion_mode(self):
        print("Releasing active Unitree motion mode before low-level control...")

        self.sport_client = SportClient()
        self.sport_client.SetTimeout(5.0)
        self.sport_client.Init()

        self.motion_switcher = MotionSwitcherClient()
        self.motion_switcher.SetTimeout(5.0)
        self.motion_switcher.Init()

        for attempt in range(12):
            status, result = self.motion_switcher.CheckMode()

            mode_name = ""
            if isinstance(result, dict):
                mode_name = result.get("name", "")

            print(
                f"MotionSwitcher CheckMode attempt {attempt}: "
                f"status={status}, result={result}"
            )

            if status == 0 and not mode_name:
                print("No active high-level motion mode remains.")
                time.sleep(0.5)
                return

            if mode_name:
                print(
                    f"Active mode '{mode_name}' detected. "
                    "Calling StandDown() and ReleaseMode()."
                )
            else:
                print(
                    "Motion mode status not clean yet. "
                    "Calling StandDown() and ReleaseMode() defensively."
                )

            try:
                self.sport_client.StandDown()
            except Exception as exc:
                print(f"Warning: SportClient.StandDown() failed: {exc}")

            time.sleep(0.2)

            try:
                self.motion_switcher.ReleaseMode()
            except Exception as exc:
                print(f"Warning: MotionSwitcher.ReleaseMode() failed: {exc}")

            time.sleep(1.0)

        status, result = self.motion_switcher.CheckMode()
        raise RuntimeError(
            "Could not release active Unitree motion mode after 12 attempts. "
            f"Last status={status}, result={result}"
        )

    def wait_for_connection(self, timeout_s=5.0):
        print("Waiting for Go2 low-state over SDK2/DDS...")

        t0 = time.time()
        while time.time() - t0 < timeout_s:
            st = self._latest_state()
            if st is not None:
                q_sum = sum(abs(st.motor_state[i].q) for i in range(12))
                if q_sum > 0.01:
                    print("Robot low-state received.")
                    self.print_state()
                    return True
            time.sleep(0.01)

        print("ERROR: no valid Go2 LowState received.")
        print(f"Check robot mode, Ethernet address, and --network={self.network}")
        return False

    def print_state(self):
        st = self._latest_state()
        if st is None:
            print("No state yet.")
            return

        print("\nCurrent joint angles, SDK order:")
        for leg in ["FR", "FL", "RR", "RL"]:
            hip = st.motor_state[self.index[f"{leg}_0"]].q
            thigh = st.motor_state[self.index[f"{leg}_1"]].q
            calf = st.motor_state[self.index[f"{leg}_2"]].q
            print(
                f"  {leg}: "
                f"hip={hip:+.3f}, thigh={thigh:+.3f}, calf={calf:+.3f}"
            )

        rpy = st.imu_state.rpy
        print(
            f"IMU: roll={rpy[0]:+.3f}, "
            f"pitch={rpy[1]:+.3f}, yaw={rpy[2]:+.3f}"
        )

    def get_state(self):
        st = self._latest_state()
        if st is None:
            raise RuntimeError("No low_state received yet")

        base_ang_vel = np.array(st.imu_state.gyroscope, dtype=np.float32)

        rpy = np.array(st.imu_state.rpy, dtype=np.float32)
        quat = quat_from_euler_xyz(rpy[0], rpy[1], rpy[2])
        projected_gravity = compute_projected_gravity(quat)

        q_sdk = np.array([st.motor_state[i].q for i in range(12)], dtype=np.float32)
        dq_sdk = np.array([st.motor_state[i].dq for i in range(12)], dtype=np.float32)

        dof_pos = q_sdk[self.config.sdk_to_train_map]
        dof_vel = dq_sdk[self.config.sdk_to_train_map]

        return base_ang_vel, projected_gravity, dof_pos, dof_vel

    def write_sdk_targets(self, target_sdk, kp, kd):
        target_sdk = np.clip(
            target_sdk,
            self.config.joint_limit_low_sdk,
            self.config.joint_limit_high_sdk,
        )

        for i, jname in enumerate(self.joint_order):
            mid = self.index[jname]
            self.low_cmd.motor_cmd[mid].mode = 0x01
            self.low_cmd.motor_cmd[mid].q = float(target_sdk[i])
            self.low_cmd.motor_cmd[mid].dq = 0.0
            self.low_cmd.motor_cmd[mid].kp = float(kp)
            self.low_cmd.motor_cmd[mid].kd = float(kd)
            self.low_cmd.motor_cmd[mid].tau = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_pub.Write(self.low_cmd)

    def send_command_train_order(self, target_train, kp, kd):
        target_sdk = target_train[self.config.train_to_sdk_map]
        self.write_sdk_targets(target_sdk, kp, kd)

    def send_damping(self, kd=6.0, n=100):
        for _ in range(n):
            for i in range(12):
                self.low_cmd.motor_cmd[i].mode = 0x01
                self.low_cmd.motor_cmd[i].q = 0.0
                self.low_cmd.motor_cmd[i].dq = 0.0
                self.low_cmd.motor_cmd[i].kp = 0.0
                self.low_cmd.motor_cmd[i].kd = float(kd)
                self.low_cmd.motor_cmd[i].tau = 0.0

            self.low_cmd.crc = self.crc.Crc(self.low_cmd)
            self.lowcmd_pub.Write(self.low_cmd)
            time.sleep(self.config.sim_dt)

    def run(self, gamepad):
        if not self.wait_for_connection():
            return

        print("\nStarting RX-00 policy on Go2 via SDK2/DDS joystick control.")
        print("Start button exits and sends damping.\n")

        motiontime = 0
        next_print_t = 0.0

        try:
            while True:
                loop_start = time.time()
                motiontime += 1
                sim_time = motiontime * self.config.sim_dt

                if gamepad.exit_requested:
                    print("\nExit requested; sending damping.")
                    self.send_damping(kd=6.0, n=200)
                    break

                base_ang_vel, projected_gravity, dof_pos, dof_vel = self.get_state()

                st = self._latest_state()
                rpy = np.array(st.imu_state.rpy, dtype=np.float32)

                if abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8:
                    print(
                        f"\nWARNING: robot tilted; "
                        f"roll={rpy[0]:.2f}, pitch={rpy[1]:.2f}"
                    )
                    self.send_damping(kd=6.0, n=200)
                    break

                if sim_time <= self.config.standup_duration:
                    rate = min(sim_time / self.config.standup_duration, 1.0)
                    self.qDes_train = (
                        dof_pos * (1.0 - rate)
                        + self.config.default_dof_pos * rate
                    )
                    self.current_qDes_sdk = self.qDes_train[self.config.train_to_sdk_map]
                    self.current_kp = self.config.kp_stand
                    self.current_kd = self.config.kd_stand

                elif sim_time <= (
                    self.config.standup_duration + self.config.stabilize_duration
                ):
                    self.qDes_train = self.config.default_dof_pos.copy()
                    self.current_qDes_sdk = self.qDes_train[self.config.train_to_sdk_map]
                    self.current_kp = self.config.kp_walk
                    self.current_kd = self.config.kd_walk

                else:
                    self.policy_counter += 1

                    if self.policy_counter >= self.policy_decimation:
                        self.policy_counter = 0

                        cmd_vx, cmd_vy, cmd_vyaw = gamepad.get_velocity()
                        commands = np.array(
                            [cmd_vx, cmd_vy, cmd_vyaw],
                            dtype=np.float32,
                        )

                        obs = build_obs(
                            base_ang_vel,
                            projected_gravity,
                            commands,
                            dof_pos,
                            dof_vel,
                            self.last_action,
                            self.config,
                        )
                        obs = normalize_obs(
                            obs,
                            getattr(self.config, "clip_observations", None),
                        )
                        obs_batch = obs[np.newaxis, :].astype(np.float32)

                        with torch.no_grad():
                            obs_tensor = torch.from_numpy(obs_batch)
                            action_tensor = self.policy(obs_tensor)
                            if isinstance(action_tensor, tuple):
                                action_tensor = action_tensor[0]
                            action = (
                                action_tensor.cpu()
                                .numpy()
                                .flatten()
                                .astype(np.float32)
                            )

                        self.last_action, self.qDes_train = action_to_joint_target(
                            action,
                            self.config,
                            self.action_filter,
                        )

                        self.current_qDes_sdk = (
                            self.qDes_train[self.config.train_to_sdk_map]
                        )
                        self.current_kp = self.config.kp_walk
                        self.current_kd = self.config.kd_walk

                if self.current_qDes_sdk is not None:
                    self.write_sdk_targets(
                        self.current_qDes_sdk,
                        self.current_kp,
                        self.current_kd,
                    )
                else:
                    self.send_damping(kd=3.0, n=1)

                if sim_time >= next_print_t:
                    vx, vy, vyaw = gamepad.get_velocity()
                    print(
                        f"t={sim_time:6.2f}s | "
                        f"cmd=({vx:+.2f}, {vy:+.2f}, {vyaw:+.2f}) | "
                        f"rpy=({rpy[0]:+.2f}, {rpy[1]:+.2f}, {rpy[2]:+.2f})"
                    )
                    next_print_t += 1.0

                elapsed = time.time() - loop_start
                sleep_time = max(0.0, self.config.sim_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            self.send_damping(kd=6.0, n=100)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="real", choices=["real", "sim"])
    parser.add_argument("--config", type=str, default="config/go2.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--network", type=str, default="enp130s0")
    args = parser.parse_args()

    if args.mode != "real":
        raise RuntimeError(
            "This replacement script is SDK2 real-robot only. "
            "Use the original repo file for --mode sim."
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(script_dir, config_path)

    config = load_config(config_path)
    policy_path = resolve_policy_path(config, args.model)

    gamepad = GamepadController(
        vx_range=config.vx_range,
        vy_range=config.vy_range,
        vyaw_range=config.vyaw_range,
    )

    print("\n" + "=" * 70)
    print(" Gamepad Control: Go2 SDK2 Real Deployment")
    print("=" * 70)
    print(" Left joystick:")
    print("   Up/Down      vx")
    print("   Left/Right   vy")
    print(" Right joystick:")
    print("   Left/Right   yaw")
    print(" D-pad:")
    print("   Up/Down      incremental vx")
    print(" Start:")
    print("   exit and send damping")
    print("=" * 70 + "\n")

    gamepad.start()

    try:
        controller = Go2SDK2JoystickController(
            config=config,
            policy_path=policy_path,
            network=args.network,
        )
        controller.run(gamepad)
    finally:
        gamepad.stop()
        print("\nProgram ended.")


if __name__ == "__main__":
    main()
