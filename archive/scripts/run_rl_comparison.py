#!/usr/bin/env python3
import os
import subprocess
import time
import sys

# Path to your scripts
SCRIPTS_DIR = os.path.expanduser("~/catkin_ws/src/my_gurdy_description/scripts")

def run_command(cmd, wait=True):
    """Run a shell command"""
    print(f"Running: {cmd}")
    process = subprocess.Popen(cmd, shell=True)
    if wait:
        process.wait()
    return process

def run_with_output(cmd):
    """Run command and get output"""
    print(f"Running: {cmd}")
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
        return output
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        print(f"Output: {e.output}")
        return None

def check_gazebo_running():
    """Check if Gazebo is running"""
    try:
        output = subprocess.check_output("rosservice list | grep gazebo", shell=True, text=True)
        return "gazebo" in output
    except:
        return False

def main():
    """Main function to run everything"""
    # 1. First, make sure no previous instances are running
    print("Killing any existing Gazebo or ROS processes...")
    run_command("killall -9 gzserver gzclient rosout rosmaster", wait=False)
    time.sleep(2)
    
    # 2. Start roscore and wait for it to be ready
    print("Starting roscore...")
    roscore_process = run_command("roscore", wait=False)
    time.sleep(5)
    
    # 3. Load the controller configuration
    print("Loading controller configuration...")
    controller_file = os.path.expanduser("~/catkin_ws/src/my_gurdy_description/config/gurdy_control.yaml")
    run_command(f"rosparam load {controller_file}", wait=True)
    
    # 4. Start Gazebo and spawn the robot
    print("Starting Gazebo...")
    gazebo_process = run_command("roslaunch gazebo_ros empty_world.launch", wait=False)
    
    # Wait for Gazebo to be fully up
    print("Waiting for Gazebo to start...")
    for _ in range(30):  # Wait up to 30 seconds
        if check_gazebo_running():
            print("Gazebo is running!")
            break
        time.sleep(1)
    else:
        print("Gazebo failed to start properly. Exiting.")
        sys.exit(1)
    
    # 5. Spawn the robot
    print("Spawning robot...")
    robot_file = os.path.expanduser("~/catkin_ws/src/my_gurdy_description/robot/gurdy.xacro")
    run_command(f"rosrun xacro xacro {robot_file} > /tmp/gurdy.urdf", wait=True)
    run_command("rosrun gazebo_ros spawn_model -file /tmp/gurdy.urdf -urdf -model gurdy -x 0.5 -y 0.0 -z 0.18", wait=True)
    
    # 6. Start robot state publisher
    print("Starting robot state publisher...")
    state_pub_process = run_command("rosrun robot_state_publisher robot_state_publisher", wait=False)
    
    # 7. Launch controllers
    print("Starting controllers...")
    run_command("rosrun controller_manager spawner --namespace=/gurdy joint_state_controller", wait=True)
    
    # Start position controllers
    controllers = [
        "head_upperlegM1_joint_position_controller",
        "head_upperlegM2_joint_position_controller", 
        "head_upperlegM3_joint_position_controller",
        "head_upperlegM4_joint_position_controller", 
        "head_upperlegM5_joint_position_controller",
        "head_upperlegM6_joint_position_controller",
        "upperlegM1_lowerlegM1_joint_position_controller", 
        "upperlegM2_lowerlegM2_joint_position_controller",
        "upperlegM3_lowerlegM3_joint_position_controller", 
        "upperlegM4_lowerlegM4_joint_position_controller",
        "upperlegM5_lowerlegM5_joint_position_controller", 
        "upperlegM6_lowerlegM6_joint_position_controller"
    ]
    
    controller_cmd = "rosrun controller_manager spawner --namespace=/gurdy " + " ".join(controllers)
    run_command(controller_cmd, wait=True)
    
    # 8. Wait for everything to be ready
    print("Waiting for controllers to be ready...")
    time.sleep(5)
    
    # 9. Create a new Python file for plotting in the scripts directory
    plotter_path = os.path.join(SCRIPTS_DIR, "simple_plotter.py")
    with open(plotter_path, "w") as f:
        f.write("""#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import rospy
from std_msgs.msg import Float32
import time

def main():
    # Initialize node
    rospy.init_node("simple_plotter")
    
    # Set up figure
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("RL Algorithm Comparison")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.grid(True)
    
    # Data storage
    algorithms = ["qlearn", "sarsa", "dqn", "policy_gradient", "ppo", "sac"]
    colors = ["blue", "green", "red", "purple", "orange", "cyan"]
    data = {algo: [] for algo in algorithms}
    lines = {}
    
    # Create lines for each algorithm
    for algo, color in zip(algorithms, colors):
        line, = ax.plot([], [], color=color, label=algo)
        lines[algo] = line
    
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)
    
    # Create subscribers
    def reward_callback(msg, algo):
        data[algo].append(msg.data)
        x = list(range(len(data[algo])))
        lines[algo].set_data(x, data[algo])
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)
    
    subscribers = {}
    for algo in algorithms:
        topic = f"/gurdy_{algo}/episode_reward"
        subscribers[algo] = rospy.Subscriber(topic, Float32, reward_callback, callback_args=algo)
    
    print("Plotter ready. Waiting for data...")
    
    # Keep alive
    try:
        while not rospy.is_shutdown():
            plt.pause(0.1)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Plotter shutting down...")
        plt.close()

if __name__ == "__main__":
    main()
""")
    
    # Make it executable
    os.chmod(plotter_path, 0o755)
    
    # 10. Start the plotter
    print("Starting the plotter...")
    plotter_process = run_command(f"cd {SCRIPTS_DIR} && python3 simple_plotter.py", wait=False)
    
    # 11. Run all the training scripts
    print("Starting all RL training scripts...")
    scripts = [
        "start_training.py",
        "train_gurdy_sarsa.py",
        "train_gurdy_dqn.py",
        "train_gurdy_policy_gradient.py",
        "train_gurdy_ppo.py",
        "train_gurdy_sac.py"
    ]
    
    processes = []
    for script in scripts:
        script_path = os.path.join(SCRIPTS_DIR, script)
        if os.path.exists(script_path):
            cmd = f"cd {SCRIPTS_DIR} && python3 {script}"
            process = run_command(cmd, wait=False)
            processes.append(process)
            time.sleep(2)  # Wait longer to make sure each script starts properly
        else:
            print(f"Warning: Script not found: {script_path}")
    
    # Keep the script running
    print("\nAll components are running. Press Ctrl+C to stop.")
    try:
        while True:
            # Check if Gazebo is still running
            if not check_gazebo_running():
                print("Warning: Gazebo is not running! Trying to restart...")
                gazebo_process = run_command("roslaunch gazebo_ros empty_world.launch", wait=False)
                time.sleep(5)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping all processes...")
        run_command("killall -9 gzserver gzclient python3", wait=False)
        print("Done.")

if __name__ == "__main__":
    main()