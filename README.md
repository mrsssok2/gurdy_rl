
in terminal 1 


```bash
cd catkin_ws
source devel/setup.bash
roslaunch my_gurdy_description one_touch_spawn_control.launch
```



in terminal 2 

```bash
pip install pyquaternion
```

```bash
cd catkin_ws

source devel/setup.bash
```

and then try out all these  one by one in the same terminal
``` bash
roslaunch my_gurdy_description train_gurdy.launch

roslaunch my_gurdy_description launch_sarsa.launch

roslaunch my_gurdy_description launch_dqn.launch

roslaunch my_gurdy_description ppo_launch.launch

roslaunch my_gurdy_description sac_launch.launch

roslaunch my_gurdy_description policy_gradient.launch

```

https://github.com/user-attachments/assets/0c3899f6-1a83-408c-ad06-21ed25311969


## Baseline

A fixed, hand-written gait run through the same walk env (`MyGurdyWalkEnv-v0`,
25 per-leg actions) the agents train on. A learned policy should beat these numbers.

| Gait | Command | Cycles | Forward distance | Falls | Notes |
|------|---------|--------|------------------|-------|-------|
| `--wave-all` | `test_gait.py 20 --wave-all` | 20 | **+0.546 m** | none | Reaches the env's 0.5 m target |
| `--wave` | `test_gait.py 20 --wave` | 20 | +0.14 m | none | Roll and pitch stay under 2° |
| `--wave --mirror` | `test_gait.py 20 --wave --mirror` | 20 | -0.037 m | none | Mirrored gait toward -x; weak |

`--wave` strokes the back legs (1, then 4) and pushes them OUT. `--wave-all` adds
the front legs (5, then 2), which pull IN. Each stroke is LIFT, return reach,
LOWER, push, so one `--wave-all` cycle is 16 env steps. Each step lasts 0.3 s of
sim time, so 20 cycles are 320 steps and 96 s of sim time.

To reproduce, start Gazebo in terminal 1 (see above). In terminal 2, load the
position controllers. The spawner keeps running, and stopping it unloads the
controllers, so leave it open:

```bash
cd catkin_ws
source devel/setup.bash
roslaunch my_gurdy_description start_controllers.launch
```

Then, in terminal 3:

```bash
cd catkin_ws
source devel/setup.bash
rosrun my_gurdy_description test_gait.py 20 --wave-all
```

The script prints the forward distance, roll and pitch after every cycle, and
the joint tracking error after every action. It ignores the env's own episode
end (stuck, goal, step limit) and stops only on a fall or after N cycles.
`test_gait.py --poses` prints the foot poses and their IK joint angles.

## Voice control demo

Speak a Hindi or Hinglish command on Windows and Gurdy walks in Gazebo. The
assistant then answers in Hinglish. The speech runs on Gnani's APIs: Prisma
v3.0 for speech to text, Timbre v2.5 (voice Poorvi, `hi-en`) for text to
speech. See `docs/gnani/notes.md`.

```
mic -> STT -> command_parser -> docker exec rostopic pub /gurdy/voice_command
     -> voice_command_node.py walks -> /gurdy/voice_feedback -> reply -> TTS -> speaker
```

Commands (the wake word "Gurdy" is optional):

| Say | Does |
|-----|------|
| "Gurdy, aage bees centimeter chalo" / "गरदी आगे बीस सेंटीमीटर चलो" | Walk 20 cm forward. Default 20 cm, at most 50 cm. Number words (das, bees, tees, pachaas, ...), digits and metres work too |
| "ruko" / "रुक जाओ" / "stop" | Stop the walk. While Gurdy walks, press Enter first to speak |
| "kitna chala" / "कितना चला" / "status" | Total distance walked, and whether Gurdy is walking |
| "RL mode" / "आर एल मोड" | Walk with the trained DQN policy |
| "normal mode" / "नॉर्मल मोड" | Walk with the scripted `--wave-all` gait (the default) |

In RL mode the node acts greedily with the DQN model. If the policy makes less
than 5 mm of progress over 32 steps, or no model is loaded, the node switches
to the scripted gait for the rest of the walk, and the reply says so.

In terminal 1 (container), start Gazebo:

```bash
cd catkin_ws
source devel/setup.bash
roslaunch my_gurdy_description one_touch_spawn_control.launch
```

In terminal 2 (container), start the position controllers and the voice
command node. `rl_model_path` is optional: absolute, or relative to the
package. The robot is reset once at startup.

```bash
cd catkin_ws
source devel/setup.bash
roslaunch my_gurdy_description voice_demo.launch rl_model_path:=<path to the saved DQN model>
```

In terminal 3 (Windows PowerShell, repo root, venv), start the client. The
container must be named `noetic`, or pass `--container <name>`. The client
runs `docker exec` itself.

```powershell
venv\Scripts\Activate.ps1
pip install -r voice_client/requirements.txt
$env:GNANI_API_KEY = "<your key>"     # read only from the environment, never saved
python voice_client/main.py
```

Press Enter, then speak for 4 s. The client prints every stage: `[TRANSCRIPT]`,
`[PARSED]`, `[COMMAND]`, `[ROBOT]` (the node's JSON result) and `[REPLY]`.
Then it plays the reply, which always starts with "Main ek AI assistant
hoon." Useful flags:

- `--type`: type the transcript instead of speaking it (no microphone or STT)
- `--no-tts`: print the reply without speaking it
- `--dry-run`: print the `docker exec` commands instead of running them
- `--mode rl`: start in RL mode
- `--echo-delay 2.0`: seconds `rostopic echo` gets to subscribe before the command goes out

To drive the node by hand from inside the container:

```bash
rostopic echo /gurdy/voice_feedback &
rostopic pub -1 /gurdy/voice_command std_msgs/String \
  "data: '{\"action\": \"walk_forward\", \"distance_m\": 0.2, \"mode\": \"scripted\"}'"
rostopic pub -1 /gurdy/voice_command std_msgs/String "data: '{\"action\": \"status\"}'"
```

`rostopic pub` reads its argument as YAML, so the JSON goes inside a
`data:` string. The node sends exactly one feedback message per command. A stop
during a walk gets none of its own: the walk's `stopped` result answers it.

Offline tests, from the repo root (no ROS needed):

```bash
python -m unittest discover -s voice_client/tests
python -m unittest discover -s scripts/tests
```






