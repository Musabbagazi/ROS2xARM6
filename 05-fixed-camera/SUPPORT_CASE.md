# xArm 6: `get_tgpio_output_digital()` deterministically triggers error 28

## Summary

On our xArm 6, **one single API call reliably puts the controller into
error 28 (End Module Communication Error)**: `get_tgpio_output_digital()`.

It fails 100% of the time, within 2–3 ms, from a known-clean state. Every
other form of communication with the end module works perfectly —
including writing the tool digital outputs, reading the tool digital
*inputs*, reading the analog inputs, and reading the end module's own
firmware version.

Because `xarm_api.get_vacuum_gripper()` internally calls
`get_tgpio_output_digital()` (via `check_on=True`), the vacuum gripper's
object-detection feature is completely unusable for us, even though the
underlying input pin reads fine when queried directly.

## System

| | |
|---|---|
| Robot | xArm 6 |
| Controller firmware | v1.6.9 |
| Controller info string | `PROTOCOL: V1, DETAIL: 6,6,XI1203,XX0000,v1.6.9, TYPE1300: [0, 0]` |
| Serial | XI120307201B99 |
| End module (tool board) firmware | **1.2.0** |
| xArm-Python-SDK | 1.18.4 |
| Tool fitted | UFACTORY xArm Vacuum Gripper, on the end-effector connector at the wrist |
| Host | Windows 11, Python 3.12, direct Ethernet to the control box |

## Measurements

Each call below was made from a verified-clean state (`clean_error()`
first, error register confirmed 0), then the error register was read
immediately afterwards. No motion at any point.

```
                                        faulted    answered
  get_tgpio_version   (tool board fw)     0/8      8   -> (0, '1.2.0')
  get_tgpio_digital   (tool input)        0/8      8   -> (0, [0,0,0,0])
  get_tgpio_analog    (tool analog in)    0/8      8   -> (0, [1.14, 1.10])
  get_tgpio_output_digital                8/8      0   -> error 28
  get_position        (no tool involved)  0/8      8
```

Writes to the tool, tested separately, 30 attempts each:

```
  set_vacuum_gripper(True)   0/30 faulted
  set_vacuum_gripper(False)  0/30 faulted
  get_vacuum_gripper()      30/30 faulted   (first fault at 3 ms)
```

`get_vacuum_gripper()` fails only because of the internal
`get_tgpio_output_digital()` call — reading the same sensor directly with
`get_tgpio_digital(ionum=0)` is clean, 8/8.

Confirmation run: the cup was commanded ON and OFF, and
`get_tgpio_digital(ionum=0)` was read 15 times in between. The controller
stayed at error 0 for the entire sequence. The vacuum pump runs normally.

## What we have ruled out

- **Not intermittent.** 100% reproducible, and instant (2–3 ms).
- **Not a latched-state artefact.** The fault is cleared and verified
  clear before every single attempt.
- **Not a wiring/connection selection mistake.** Tested with
  `hardware_version=1` (tool GPIO 0/1) and `hardware_version=2` (3/4).
- **Not power.** Joint rails read 23.7 V; the pump draws current and runs
  normally on the same connector.
- **Not a power-up state.** Survives an e-stop power cycle of the control
  box and a full arm power-off/on. Reproduced on a fresh boot.
- **Not load-related.** The fault does not need the pump running — the
  call fails with the cup off and idle.
- **Not caused by our polling.** The controller stays at error 0
  indefinitely when the tool is left alone (verified over 30 s with the
  pump running on a single command, and repeatedly at idle).

## Questions

1. Is `get_tgpio_output_digital()` **supported on end module firmware
   1.2.0** with controller v1.6.9? Our results look like an unsupported
   or mismatched command being answered with a communication error rather
   than a "not supported" code.
2. If it is a firmware gap, **what end module firmware should we be on**,
   and what is the correct procedure to update the tool board?
3. Is there a **supported way to read the vacuum gripper's
   object-detection state** that avoids this call? Reading
   `get_tgpio_digital(ionum=0)` directly works for us — is that the
   correct pin and the correct interpretation (0 = not picked,
   1 = picked) for the xArm Vacuum Gripper?
4. Should `xarm_api.get_vacuum_gripper()` be calling
   `get_tgpio_output_digital()` at all? It appears to use it only to
   decide whether to report "off"; on our system that turns a working
   sensor read into a controller fault.

## Reproducing it

```python
from xarm.wrapper import XArmAPI
arm = XArmAPI("192.168.1.197")
arm.clean_error(); arm.clean_warn()
arm.motion_enable(True); arm.set_mode(0); arm.set_state(0)
print("before:", arm.get_err_warn_code())     # (0, [0, 0])
arm.get_tgpio_output_digital()
print("after :", arm.get_err_warn_code())     # (0, [28, 0])
```
