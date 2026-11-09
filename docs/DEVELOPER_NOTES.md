
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
| `--wave-all --mirror` | `test_gait.py 20 --wave-all --mirror` | 20 | **0.583 m backward** | none | 1.82 mm/step; y drift 0.5 cm |
| `--left` | `test_gait.py 10 --left` | 10 | 0.259 m toward +y | none | 1.08 mm/step; x drift 0.6 cm |
| `--right` | `test_gait.py 10 --right` | 10 | 0.292 m toward -y | none | 1.22 mm/step; x drift 0.4 cm |

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

Gurdy is front/back symmetric, so `test_gait.py 20 --wave-all --mirror` is the
backward walk: the front legs (2, then 5) push OUT and the back legs (4, then
1) pull IN. With `--mirror` the script reports the backward distance and the
y drift. Unlike `--wave --mirror`, it matches the forward gait: 0.583 m in
320 steps (Baseline), so `voice_walk.BACKWARD_M_PER_STEP` is 1.8 mm per step.

Gurdy's feet move only along their radial spokes, so it cannot turn, but it
can step sideways. `test_gait.py 10 --left` walks toward +y: leg 6 (270°)
pushes OUT, leg 3 (90°) pulls IN, then the diagonal pairs 1 OUT / 4 IN and
2 OUT / 5 IN, whose x components cancel. That is 24 env steps per cycle.
`--right` mirrors it across the x axis toward -y. For these the script reports
the sideways distance (world y) and the x drift. Measured (Baseline), it makes
1.08 mm per step left and 1.22 mm per step right; `voice_walk.SIDE_M_PER_STEP`
uses the slower 1.1 mm so step limits and timeouts are safe both ways.

## Evaluating DQN checkpoints

`evaluate.py` runs saved DQN models greedily (epsilon 0) through the same walk env
and episode end as training (goal, fall, stuck, or `nsteps`). For each episode
it prints the forward distance, steps, how the episode ended, falls and invalid
lifts, then a summary. Training saves a checkpoint every `save_model_freq`
episodes, as `output/dqn/gurdy_dqn_model_<episode>`.

Start Gazebo (terminal 1) and the controllers (`start_controllers.launch`,
terminal 2) as in Baseline. Training must not be running, because the two
would share one simulation. Then, in terminal 3:

```bash
cd catkin_ws
source devel/setup.bash
# one checkpoint, 5 episodes
rosrun my_gurdy_description evaluate.py output/dqn/gurdy_dqn_model_100 --episodes 5
# every checkpoint in output/dqn, 3 episodes each, ranked best first
rosrun my_gurdy_description evaluate.py --all
# also save every episode as CSV
rosrun my_gurdy_description evaluate.py --all --episodes 5 --csv output/dqn/eval.csv
```

The ranking sorts by mean forward distance, then fewer falls, then fewer
steps. Compare it with the Baseline: the scripted `--wave-all` gait reaches
the 0.5 m goal in about 290 steps. Paths are relative to the package.

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
| "Gurdy, baayen bees centimeter chalo" / "गरदी बाएं बीस सेंटीमीटर चलो" | Step 20 cm left (+y) with the scripted side-step gait (no RL). Same distances and 50 cm cap |
| "daayen tees cm chalo" / "दाएँ तीस सेंटीमीटर चलो" | Step 30 cm right (-y) |
| "peeche bees cm chalo" / "पीछे बीस सेंटीमीटर चलो" / "go back" | Walk 20 cm backward with the mirrored `--wave-all` gait (no RL). Same distances and 50 cm cap |
| "ruko" / "रुक जाओ" / "stop" | Stop the walk. While Gurdy walks, press Enter first to speak |
| "kitna chala" / "कितना चला" / "status" | Total distance walked, and whether Gurdy is walking |
| "RL mode" / "आर एल मोड" | Walk with the trained DQN policy |
| "normal mode" / "नॉर्मल मोड" | Walk with the scripted `--wave-all` gait (the default) |

In RL mode the node acts greedily with the DQN model. If the policy makes less
than 5 mm of progress over 32 steps, or no model is loaded, the node switches
to the scripted gait for the rest of the walk, and the reply says so.

A sentence with a direction Gurdy cannot walk (upar, neeche), a turn (mudo,
ghoomo, turn), or two directions ("aage baayen", "aage peeche") is not
understood rather than treated as a forward walk; the reply lists what works.

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
- `--ui`: also talk from the dashboard's Speak button (see below); Enter still works

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

### Live dashboard

`dashboard.py` opens a window for the demo video. It needs a display; WSLg
works, as for the training plots. The left side is a top-down map in metres,
from `/gurdy/odom`. It shows a grid, the start point, the path trail, Gurdy's
position with a heading arrow, and the start and target point of the running
walk. The right side refreshes at about 10 Hz and shows:

- position, heading and roll / pitch / height
- this walk's distance (done / target) and the total
- the lifted leg, read from the position controller commands
- status (STANDING, WALKING, STOPPED, FALLEN) and mode (RL / scripted)
- the last transcript, command and result from the voice topics

Start it in another container terminal while `voice_demo.launch` runs:

```bash
cd catkin_ws
source devel/setup.bash
rosrun my_gurdy_description dashboard.py
```

#### Speak and Stop buttons

The bottom of the dashboard has a big **Speak** button and a red **Stop**
button, the voice client's state (IDLE / LISTENING / THINKING / ROBOT
ACTING / SPEAKING) and its last spoken reply. The dashboard runs in the
container and the mic is on Windows, so the buttons talk to the client over
two topics:

```
Speak -> /gurdy/ui/talk  {"id": ...}  -> main.py --ui: record 4 s, STT, parse, robot, TTS
main.py --ui -> /gurdy/ui/state {"state": "LISTENING", "seq": 3, "reply": ...} -> dashboard
Stop  -> /gurdy/voice_command {"action": "stop"}   (straight to the robot, no voice)
```

Start the client with `--ui`. It keeps one `rostopic echo /gurdy/ui/talk`
open and restarts it if it dies. While Gurdy walks, Speak records and only
"ruko" (stop) or "wapas" (reset) is accepted, as with Enter. Presses made
while the client is busy thinking or speaking are dropped. If the client
doesn't answer a press within 8 s, the dashboard shows NO VOICE CLIENT.
`_buttons:=false` hides the strip.

```powershell
python voice_client/main.py --ui
```

The path is logged at 10 Hz to `output/demo/path_<timestamp>.csv` in the package.
Plot a saved log as a PNG, with the path coloured by status and the walk
targets and falls marked. This needs only matplotlib:

```bash
python3 scripts/plot_path.py output/demo/path_20261109_101500.csv   # writes path_20261109_101500.png
```

The transcript needs a font with Devanagari. The dashboard looks for Noto
Sans Devanagari, Lohit Devanagari and others, and logs a warning if it finds
none. In that case install one in the container, e.g.
`apt-get install fonts-noto-core`, then delete the matplotlib font cache
(`rm -rf ~/.cache/matplotlib`).

Offline tests, from the repo root (no ROS needed):

```bash
python -m unittest discover -s voice_client/tests
python -m unittest discover -s scripts/tests
```

The vision tests (`scripts/tests/test_vision.py`) need OpenCV and are skipped
without it; run them in the container, where `python3` has cv2.

### Web dashboard (browser)

`web_dashboard.py` serves the same view as `dashboard.py` to a browser on
Windows, at http://localhost:8080. It's handier than the Tk window for a screen
recording and fits 1920x1080. The page has these parts:

- a top-down map with the path, target and heading
- the live annotated camera as MJPEG
- status, position, heading, distances, roll / pitch, the lifted leg and the mode
- the last transcript, command and result
- the voice client's state and last reply
- the big **Speak** and red **Stop** buttons. The keys Space and Esc do the same.

It updates over Server-Sent Events at 10 Hz. It uses the same
`dashboard_state.py` logic and topics as the Tk dashboard and also logs the
path to `output/demo/`. Both dashboards can run at once.

It needs Flask in the container, once (see `requirements-web.txt`):

```bash
pip3 install -r ~/catkin_ws/src/my_gurdy_description/requirements-web.txt
```

Run it in a container terminal while `voice_demo.launch` runs, then open
http://localhost:8080 in Chrome on Windows:

```bash
cd catkin_ws
source devel/setup.bash
rosrun my_gurdy_description web_dashboard.py
# _port:=8081 another port (publish it too), _log:=false no CSV log
```

The `noetic` container must publish port 8080. It was created without published
ports, so it was recreated from a snapshot with the same mounts and
environment plus `-p 8080:8080`. Run this from the Ubuntu WSL shell so the
`/home/mahadev/...` and `/mnt/wslg` paths resolve as before:

```bash
docker commit noetic noetic_snapshot
docker rename noetic noetic_old && docker stop noetic_old
docker run -dit --name noetic -p 8080:8080 \
  -v /home/mahadev/catkin_ws:/root/catkin_ws \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg \
  -e DISPLAY=:0 -e WAYLAND_DISPLAY=wayland-0 -e XDG_RUNTIME_DIR=/run/user/1000 \
  noetic_snapshot bash
```

`docker exec -it noetic bash` opens terminals as before. To roll back to the
old container, run `docker stop noetic && docker rename noetic noetic_web &&
docker rename noetic_old noetic && docker start noetic`. Once the new container
has proved itself, delete the backup with `docker rm noetic_old && docker rmi
noetic_snapshot`.

## Computer vision demo

Gurdy's head camera looks forward along its walking direction (+x), tilted
12° down. `vision_node.py` finds red, green and blue objects in its image,
and two voice commands use them: "kya dikh raha hai" describes what Gurdy
sees, and "laal gend ke paas jao" walks up to an object.

```
/gurdy/camera/image_raw -> vision_node.py -> /gurdy/detections (JSON)
                                          -> /gurdy/camera/annotated (image)
/gurdy/detections -> voice_command_node.py: describe, go_to_object
```

| Say | Does |
|-----|------|
| "kya dikh raha hai" / "क्या दिख रहा है" / "what do you see" | Lists the objects in view with their side, e.g. "Mujhe saamne laal gend aur baayen taraf neela dabba dikh raha hai" |
| "laal gend ke paas jao" / "लाल गेंद के पास जाओ" | Walks up to the red ball: "Main laal gend ke paas pahunch gaya" |
| "neele dabbe tak jao" / "hara dabba" / "go to the green box" | The same for a box. Colours: laal / neela / hara (red, blue, green). Objects: gend / ball, dabba / box, or none for any shape |
| "ruko" | Stops the approach, as for a walk |

If the object isn't in view, the reply is "Mujhe laal gend nahi dikh rahi".
A colour alone ("laal") is not understood. The command needs an object
word, a go word (jao, chalo, paas, tak), or both.

### Running it

Terminal 1 (container): Gazebo with the demo world. The red ball (6 cm) is
0.35 m straight ahead, the blue box (8 cm) 0.45 m ahead and 22 cm left, and
the green box 0.5 m ahead and 25 cm right. All three are static.

```bash
cd catkin_ws
source devel/setup.bash
roslaunch my_gurdy_description one_touch_spawn_control.launch world:=demo
# gui:=false runs Gazebo without its window
```

Terminal 2 (container): controllers, the voice command node and the vision
node, which `voice_demo.launch` now starts by default (`vision:=false`
leaves it out).

```bash
cd catkin_ws
source devel/setup.bash
roslaunch my_gurdy_description voice_demo.launch
```

Terminal 3 (container, optional): the dashboard, now with the live
annotated camera view in the middle (`_camera:=false` hides it).

```bash
cd catkin_ws
source devel/setup.bash
rosrun my_gurdy_description dashboard.py
```

Terminal 4 (Windows PowerShell, repo root, venv): the voice client, as
above. Use `--type --no-tts` to type the commands instead.

```powershell
venv\Scripts\Activate.ps1
$env:GNANI_API_KEY = "<your key>"
python voice_client/main.py
```

By hand, inside the container:

```bash
rostopic echo /gurdy/detections            # what the camera sees, ~8-10 Hz
rosrun rqt_image_view rqt_image_view /gurdy/camera/annotated
rostopic echo /gurdy/voice_feedback &
rostopic pub -1 /gurdy/voice_command std_msgs/String "data: '{\"action\": \"describe\"}'"
rostopic pub -1 /gurdy/voice_command std_msgs/String \
  "data: '{\"action\": \"go_to_object\", \"color\": \"red\", \"shape\": \"ball\"}'"
```

### How it works

`scripts/vision.py` (no ROS) thresholds the image in HSV for each colour,
removes specks with a morphological opening, and keeps each blob over
`MIN_AREA_FRACTION` of the image. The opening also removes the thin red line
Gazebo draws along the world x axis. Each detection has a pixel centroid
and box, and a **bearing** from the image centre (positive is left, from
the 80° field of view). It also has an **area fraction**, and a side
(left / center / right, centre is within ±10°). The **shape** guess is
"ball" if the blob fills most of its enclosing circle. A blob cut off by
the image border is a ball if its free edge fits a circle. The **distance**
is rough. The blob's lowest pixel is where it meets the ground, so the
camera height (0.128 m) and tilt give the distance. If the bottom of the
image cuts the blob off, the distance comes from its width and known size
instead. On a real frame of the demo world, the bearings were within 1-2°
and the distances within 4 cm of the world file.

`scripts/visual_approach.py` (no ROS) runs `go_to_object`. Gurdy can't
turn, so it lines up by stepping sideways:

1. Wait for a camera frame taken after the last move. Take the largest
   detection of the colour, and of the shape if one was asked for.
2. If it isn't there, count a miss. After `lost_checks` misses in a row,
   stop: "not_found" if it was never seen, else "lost".
3. Stop as "reached" if the target covers `close_area` of the image and is
   within `bearing_tol_deg` of the centre. If it is close but off-centre,
   side-step first (step 4). A target cut off by the bottom of the image
   also counts as reached: the bottom row sees the ground ~0.13 m ahead,
   so a cut-off object is closer than that.
4. If it is more than `bearing_tol_deg` off centre, side-step toward it for
   1-3 cycles (~2.6 cm each), sized by its estimated sideways offset.
   Otherwise walk forward 1 cycle (~2.7 cm), or 2 while it is still small.

Every move is whole cycles of the existing scripted gaits, so all legs are
down between moves. A fall, a stop, and 3 moves in a row without progress
("stuck") end the approach as they end a walk. So do 40 moves or 800 steps
("gave_up").

Measured in Gazebo through the voice client:

| Command | Result |
|---------|--------|
| laal gend ke paas jao | reached, 6 moves, 104 steps, 1 side-step; stopped ~0.16 m from the ball, 1.6° off centre |
| neele dabbe ke paas jao (29° left) | reached, 5 moves, 264 steps, all side-steps; the last one because the box was close but 12.4° off, so it stopped at 5.3° |
| hare dabbe ke paas jao (29° right) | reached, 13 moves, 384 steps |
| hari gend ke paas jao | not_found, no steps ("Mujhe hari gend nahi dikh rahi") |
| ruko during an approach | stopped |

### What to tune

- **HSV ranges** (`vision.HSV_RANGES`, or the node's `~hsv_ranges`
  param). OpenCV hue runs 0-179, and red wraps around 0, so it has two
  ranges. Saturation ≥ 100 keeps out the grey ground and sky. Value ≥ 40
  keeps the shaded side of an object. To check a colour, save a frame and
  look at its HSV values (`cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`). For
  real-world lighting, raise the saturation and value minimums first.
  Example override:
  `rosrun my_gurdy_description vision_node.py _hsv_ranges:='{"red": [[[0, 120, 60], [10, 255, 255]], [[170, 120, 60], [179, 255, 255]]]}'`
- **Close threshold** (`close_area_ball`, default 0.04, and
  `close_area_box`, default 0.09, args of `voice_demo.launch`). A larger
  value stops closer. Keep it below the area the object reaches before the
  bottom of the image cuts it off, from ~0.15 m. In Gazebo the ball's area
  peaked at ~0.056, so with 0.06 Gurdy never stopped and walked into the
  ball. 0.04 stops a 6 cm ball ~0.18 m from the camera. The box is bigger,
  so it needs a larger fraction for the same distance. The cut-off rule in
  step 3 is the backstop.
- **Bearing tolerance** (`bearing_tol_deg`, default 8). Smaller lines up
  more precisely but side-steps more. A fixed sideways offset looks bigger
  as Gurdy gets closer: 3 cm is 6° at 0.3 m but 11° at 0.15 m.
- **lost_checks** (default 3): frames in a row without the target before
  giving up.
- **min_area_fraction** (`~min_area_fraction`, default 0.0005): raise it
  if noise or far-off clutter is detected.
- **Camera tilt and height.** `robot/gurdy.xacro` (`camera_tilt_rad`, the
  `head_camera_joint` origin) and `vision_node.py` (`~camera_tilt_deg`,
  `~camera_height_m`) must agree, or the distances are off.

Example: `roslaunch my_gurdy_description voice_demo.launch close_area_ball:=0.05 bearing_tol_deg:=6`

Limits: Gurdy cannot turn, so objects more than ~40° to the side are out of
view. The side-step gait also turns the body slowly, by about 3° per cycle
and clockwise in both directions: about −30° after 10 cycles. The approach
doesn't mind, because bearings are measured in the camera frame, but after
a long sideways approach Gurdy no longer faces +x.
