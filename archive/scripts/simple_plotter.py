#!/usr/bin/env python3
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
