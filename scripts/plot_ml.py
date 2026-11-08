import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class QLearningVisualizer:
    def __init__(self, nepisodes=1000):
        self.nepisodes = nepisodes
        self.rewards = []
        self.epsilons = []

    def generate_training_data(self):
        """
        Simulate Q-learning training data
        """
        epsilon = 1.0
        epsilon_discount = 0.9

        for episode in range(self.nepisodes):
            # Simulate reward with some randomness
            reward = np.random.normal(5, 2)
            self.rewards.append(reward)

            # Simulate epsilon decay
            epsilon = max(0.05, epsilon * epsilon_discount)
            self.epsilons.append(epsilon)

    def plot_metrics(self):
        """
        Create multi-plot visualization of training metrics
        """
        plt.figure(figsize=(12, 8))

        # Rewards Plot
        plt.subplot(2, 2, 1)
        plt.plot(range(1, self.nepisodes + 1), self.rewards, 'b-')
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')

        # Cumulative Rewards Plot
        plt.subplot(2, 2, 2)
        plt.plot(range(1, self.nepisodes + 1), np.cumsum(self.rewards), 'g-')
        plt.title('Cumulative Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')

        # Epsilon Decay Plot
        plt.subplot(2, 2, 3)
        plt.plot(range(1, self.nepisodes + 1), self.epsilons, 'r-')
        plt.title('Exploration Rate (Epsilon) Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')

        # Running Average of Rewards
        plt.subplot(2, 2, 4)
        window_size = 50
        running_avg = np.convolve(self.rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(1, len(running_avg) + 1), running_avg, 'm-')
        plt.title('Running Average of Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')

        plt.tight_layout()
        plt.show()

def main():
    # Initialize ROS node
    rospy.init_node('q_learning_visualizer', anonymous=True)
    
    # Create visualizer
    visualizer = QLearningVisualizer(nepisodes=1000)
    
    # Generate training data
    visualizer.generate_training_data()
    
    # Plot metrics
    visualizer.plot_metrics()

if __name__ == '__main__':
    main()