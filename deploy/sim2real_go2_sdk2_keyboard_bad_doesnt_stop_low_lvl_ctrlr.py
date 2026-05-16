#!/usr/bin/env python3

import argparse
import os
import sys
import time
import threading

import numpy as np
import torch

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from action_lpf import ActionTargetLowPassFilter, action_to_joint_target
from sim2sim2real_keyboard import (
    KeyboardController,
    build_obs_45,
    compute_projected_gravity,
    load_config,
    normalize_obs,
    quat_from_euler_xyz,
)

POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0


class Go2SDK2Controller:
    """
    RX-00 policy loop with Unitree Go2 SDK2/DDS transport.

    Training order is handled by config.train_to_sdk_map and config.sdk_to_train_map.
    SDK motor order is Unitree Go2 order:
      FR_0, FR_1, FR_2, FL_0, FL_1, FL_2, RR_0, RR_1, RR_2, RL_0, RL_1, RL_2
    """

    def __init__(self, config, policy_path, network):
        self.config = config
        self.network = network

        self.d = {
            "FR_0": 0, "FR_1": 1, "FR_2": 2,
            "FL_0": 3, "FL_1": 4, "FL_2": 5,
            "RR_0": 6, "RR_1": 7, "RR_2": 8,
            "RL_0": 9, "RL_1": 10, "RL_2": 11,
        }
        self.joint_order = [
            "FR_0", "FR_1", "FR_2",
            "FL_0", "FL_1", "FL_2",
            "RR_0", "RR_1", "RR_2",
            "RL_0", "RL_1", "RL_2",
        ]

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

        print("Go2 SDK2 controller initialized")

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

        print("\nCurrent joint angles, SDK order FR, FL, RR, RL:")
        for leg in ["FR", "FL", "RR", "RL"]:
            hip = st.motor_state[self.d[f"{leg}_0"]].q
            thigh = st.motor_state[self.d[f"{leg}_1"]].q
            calf = st.motor_state[self.d[f"{leg}_2"]].q
            print(f"  {leg}: hip={hip:+.3f}, thigh={thigh:+.3f}, calf={calf:+.3f}")

        rpy = st.imu_state.rpy
        print(f"IMU: roll={rpy[0]:+.3f}, pitch={rpy[1]:+.3f}, yaw={rpy[2]:+.3f}")

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
            mid = self.d[jname]
            self.low_cmd.motor_cmd[mid].mode = 0x01
            self.low_cmd.motor_cmd[mid].q = float(target_sdk[i])
            self.low_cmd.motor_cmd[mid].dq = 0.0
            self.low_cmd.motor_cmd[mid].kp = float(kp)
            self.low_cmd.motor_cmd[mid].kd = float(kd)
            self.low_cmd.motor_cmd[mid].tau = 0.0

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_pub.Write(self.low_cmd)

    def send_command(self, target_train, kp, kd):
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

    def run(self, keyboard):
        if not self.wait_for_connection():
            return

        print("\nStarting RX-00 policy on Go2 via SDK2/DDS.")
        print("Press SPACE to zero velocity command. Press Q or ESC to exit with damping.\n")

        motiontime = 0
        next_print_t = 0.0

        try:
            while True:
                loop_t0 = time.time()
                motiontime += 1
                sim_time = motiontime * self.config.sim_dt

                if keyboard.exit_requested:
                    print("\nExit requested; sending damping.")
                    self.send_damping(kd=6.0, n=200)
                    break

                base_ang_vel, projected_gravity, dof_pos, dof_vel = self.get_state()

                rpy = np.array(self._latest_state().imu_state.rpy, dtype=np.float32)
                if abs(rpy[0]) > 0.8 or abs(rpy[1]) > 0.8:
                    print(f"\nWARNING: robot tilted; roll={rpy[0]:.2f}, pitch={rpy[1]:.2f}")
                    self.send_damping(kd=6.0, n=200)
                    break

                # Phase 1: interpolate from current pose to default pose.
                if sim_time <= self.config.standup_duration:
                    rate = min(sim_time / self.config.standup_duration, 1.0)
                    self.qDes_train = dof_pos * (1.0 - rate) + self.config.default_dof_pos * rate
                    self.send_command(self.qDes_train, self.config.kp_stand, self.config.kd_stand)

                # Phase 2: hold default pose.
                elif sim_time <= self.config.standup_duration + self.config.stabilize_duration:
                    self.qDes_train = self.config.default_dof_pos.copy()
                    self.send_command(self.qDes_train, self.config.kp_walk, self.config.kd_walk)

                # Phase 3: policy.
                else:
                    self.policy_counter += 1

                    if self.policy_counter >= self.policy_decimation:
                        self.policy_counter = 0

                        cmd_vx, cmd_vy, cmd_vyaw = keyboard.get_velocity()
                        commands = np.array([cmd_vx, cmd_vy, cmd_vyaw], dtype=np.float32)

                        obs = build_obs_45(
                            base_ang_vel,
                            projected_gravity,
                            commands,
                            dof_pos,
                            dof_vel,
                            self.last_action,
                            self.config,
                        )
                        obs = normalize_obs(obs, self.config.clip_observations)
                        obs_batch = obs[np.newaxis, :].astype(np.float32)

                        with torch.no_grad():
                            obs_tensor = torch.from_numpy(obs_batch)
                            action_tensor = self.policy(obs_tensor)
                            if isinstance(action_tensor, tuple):
                                action_tensor = action_tensor[0]
                            action = action_tensor.cpu().numpy().flatten().astype(np.float32)

                        self.last_action, self.qDes_train = action_to_joint_target(
                            action,
                            self.config,
                            self.action_filter,
                        )

                    # Publish at sim_dt rate; update targets at policy rate.
                    self.send_command(self.qDes_train, self.config.kp_walk, self.config.kd_walk)

                if sim_time >= next_print_t:
                    vx, vy, vyaw = keyboard.get_velocity()
                    print(
                        f"t={sim_time:6.2f}s | "
                        f"cmd=({vx:+.2f}, {vy:+.2f}, {vyaw:+.2f}) | "
                        f"rpy=({rpy[0]:+.2f}, {rpy[1]:+.2f}, {rpy[2]:+.2f})"
                    )
                    next_print_t += 1.0

                sleep_t = self.config.sim_dt - (time.time() - loop_t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)

        finally:
            self.send_damping(kd=6.0, n=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", required=True, help="Robot Ethernet interface, e.g. enp130s0")
    parser.add_argument("--config", default="config/go2.yaml")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(script_dir, args.config)
    config = load_config(config_path)

    if args.model:
        policy_path = os.path.join(os.path.dirname(config.policy_path), args.model)
    else:
        policy_path = config.policy_path

    keyboard = KeyboardController(
        vx_range=config.vx_range,
        vy_range=config.vy_range,
        vyaw_range=config.vyaw_range,
    )

    keyboard.start()
    try:
        controller = Go2SDK2Controller(config, policy_path, args.network)
        controller.run(keyboard)
    finally:
        keyboard.stop()
        print("\nProgram ended.")


if __name__ == "__main__":
    main()
