# RX-00 `legged_rl_gym` Go2 Sim2Real Deployment Notes

## Summary

The original RX-00 real deployment command:

```bash
python sim2sim2real_keyboard.py --mode real --config config/go2.yaml
```

uses the legacy `unitree_legged_sdk` UDP backend. On this Go2 setup, that path failed because the container could import `robot_interface`, but UDP low-state reception returned all-zero joint states:

```text
q_sum: 0.0
No valid joint data received
```

The successful path was to keep RX-00's policy / observation / action logic, but replace the real-robot transport with host-side Unitree SDK2 + CycloneDDS.

## Host setup

Use the host machine for robot communication, not the Docker container.

Example conda environment:

```bash
conda create -n amp-go2-sdk2 python=3.10 -y
conda activate amp-go2-sdk2

python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy scipy pyyaml
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install SDK2 Python bindings:

```bash
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
python -m pip install -e .
```

If CycloneDDS is not found during install, set:

```bash
export CYCLONEDDS_HOME=/path/to/cyclonedds/install
```

## Network setup

The Go2 Ethernet interface used here was:

```bash
enp130s0
```

Configure the host-side robot interface:

```bash
sudo ip link set enp130s0 up
sudo ip addr replace 192.168.123.222/24 dev enp130s0
ip route get 192.168.123.10
```

The route should show `dev enp130s0`.

## Key fix

The first SDK2 Python controller caused vibration when run directly, but worked after running Unitree's `go2_ctrl` first. This indicated that `go2_ctrl` was performing required SDK2 mode takeover / release.

The fix was to add the Unitree SDK2 low-level takeover sequence to the custom RX-00 SDK2 deployment scripts:

```python
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient
```

Before publishing low-level `rt/lowcmd`, the script must:

1. initialize `SportClient`;
2. initialize `MotionSwitcherClient`;
3. call `MotionSwitcherClient.CheckMode()`;
4. if a high-level mode is active, call `SportClient.StandDown()`;
5. call `MotionSwitcherClient.ReleaseMode()`;
6. only then begin low-level policy control.

This avoids fighting the active Unitree sport/high-level controller.

## Files added/replaced

Two SDK2-based replacement scripts were created in:

```text
deploy/
```

### Keyboard version

```text
deploy/sim2real_go2_sdk2_keyboard.py
```

Run with:

```bash
cd ~/AMP_docker_env/legged_rl_gym/deploy
conda activate amp-go2-sdk2

python sim2real_go2_sdk2_keyboard.py \
  --network enp130s0 \
  --config config/go2.yaml
```

### Joystick/controller version

The joystick version replaced:

```text
deploy/sim2sim2real_joystick.py
```

Run with:

```bash
cd ~/AMP_docker_env/legged_rl_gym/deploy
conda activate amp-go2-sdk2

python sim2sim2real_joystick.py \
  --mode real \
  --config config/go2.yaml \
  --network enp130s0
```

The default network argument was set to `enp130s0`, so this also works:

```bash
python sim2sim2real_joystick.py --mode real --config config/go2.yaml
```

## Policy file

The config expects:

```text
deploy/exported_policy/go2/policy_1.pt
```

Check before running hardware:

```bash
ls -lh ~/AMP_docker_env/legged_rl_gym/deploy/exported_policy/go2/policy_1.pt
```

If needed, pass a different file name:

```bash
python sim2real_go2_sdk2_keyboard.py \
  --network enp130s0 \
  --config config/go2.yaml \
  --model policy.pt
```

## Safety checks before each run

Stop other controllers first:

```bash
pkill -f go2_ctrl || true
pkill -f sim2real || true
```

Confirm SDK2 low-state is visible before commanding the robot:

```bash
python - <<'PY'
import time, threading, sys
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

state = {"msg": None}
lock = threading.Lock()

def cb(msg):
    with lock:
        state["msg"] = msg

ChannelFactoryInitialize(0, "enp130s0")
sub = ChannelSubscriber("rt/lowstate", LowState_)
sub.Init(cb, 10)

t0 = time.time()
while time.time() - t0 < 5.0:
    with lock:
        msg = state["msg"]
    if msg is not None:
        q = [msg.motor_state[i].q for i in range(12)]
        print("q:", q)
        print("q_sum:", sum(abs(x) for x in q))
        sys.exit(0)
    time.sleep(0.01)

print("ERROR: no LowState received")
sys.exit(1)
PY
```

Do not proceed if `q_sum` is zero or no low-state is received.

## Important notes

- Do not run `go2_ctrl` and the RX-00 SDK2 deployment script at the same time.
- The original RX-00 UDP command does not use host `unitree_sdk2`; it uses `unitree_legged_sdk`.
- The Docker container remains useful for training/exporting the RX-00 policy.
- Real robot communication should be done on the host through SDK2/CycloneDDS.
- The SDK2 scripts retain RX-00's policy, observation construction, action filtering, joint mapping, and `go2.yaml` configuration, but replace the transport layer.
