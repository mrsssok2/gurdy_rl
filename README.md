
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

roslaunch my_gurdy_description ppo_launch.launch

```

https://github.com/user-attachments/assets/0c3899f6-1a83-408c-ad06-21ed25311969






