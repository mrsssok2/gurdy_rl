
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






