#!/usr/bin/env python
"""
Algorithm Manager for Reinforcement Learning

This module manages multiple RL algorithms running in parallel, providing
a unified interface for training, visualization, and comparison.
"""
import rospy
import os
import threading
import time
import json
from algorithm_wrapper import get_algorithm_wrapper

class AlgorithmManager:
    """Manages multiple RL algorithms running in parallel"""
    def __init__(self, algorithms, config, visualization=None):
        """
        Initialize the algorithm manager.
        
        Args:
            algorithms (dict): Dictionary of enabled algorithms and their configurations
            config (dict): Global configuration dictionary
            visualization (RLVisualization, optional): Visualization object
        """
        self.config = config
        self.visualization = visualization
        self.algorithms = {}
        self.threads = {}
        self.results = {}
        
        # Initialize algorithm wrappers
        for algo_name, algo_config in algorithms.items():
            # Create namespace if specified
            namespace = f"{algo_name}_ns" if algo_config.get('namespaced', True) else None
            
            # Create algorithm wrapper
            self.algorithms[algo_name] = get_algorithm_wrapper(
                algo_name, config, namespace)
            
            rospy.loginfo(f"Initialized {algo_name} algorithm")
        
        # Create output directories
        os.makedirs(config['training']['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['training']['output_dir'], 'results'), exist_ok=True)
        
        # Load models if specified
        if config['training'].get('load_models', False):
            self._load_models()
    
    def _load_models(self):
        """Load models for all algorithms if available"""
        for algo_name, algo in self.algorithms.items():
            # Try to find latest model
            model_dir = os.path.join(self.config['training']['output_dir'], algo_name)
            if not os.path.exists(model_dir):
                continue
                
            # Find the best model or the latest one
            model_files = [f for f in os.listdir(model_dir) 
                         if f.endswith('.h5') or f.endswith('.pkl')]
            
            best_model = None
            for f in model_files:
                if '_best' in f:
                    best_model = os.path.join(model_dir, f)
                    break
            
            if best_model:
                algo.load_model(best_model)
            elif model_files:
                # Sort by episode number if available
                episode_models = {}
                for f in model_files:
                    parts = f.split('_')
                    if len(parts) > 1:
                        try:
                            ep_num = int(parts[-1].split('.')[0])
                            episode_models[ep_num] = f
                        except ValueError:
                            pass
                
                if episode_models:
                    latest_ep = max(episode_models.keys())
                    latest_model = os.path.join(model_dir, episode_models[latest_ep])
                    algo.load_model(latest_model)
    
    def _train_algorithm(self, algo_name):
        """
        Train a specific algorithm in a separate thread.
        
        Args:
            algo_name (str): Name of algorithm to train
        """
        try:
            rospy.loginfo(f"Starting training for {algo_name}")
            
            algo = self.algorithms[algo_name]
            episodes = self.config['training']['episodes']
            max_steps = self.config['environment']['max_steps']
            
            # Define callback for visualization updates
            def update_callback(name, episode, reward, q_value, height, fall, time):
                if self.visualization:
                    self.visualization.update_data(
                        name, episode, reward, q_value, height, fall, time)
            
            # Run training
            results = algo.train(episodes, max_steps, update_callback)
            
            # Store results
            self.results[algo_name] = results
            
            rospy.loginfo(f"Completed training for {algo_name}")
            
        except Exception as e:
            rospy.logerr(f"Error training {algo_name}: {e}")
    
    def run_algorithm(self, algo_name):
        """
        Run a specific algorithm in a separate thread.
        
        Args:
            algo_name (str): Name of algorithm to run
            
        Returns:
            threading.Thread: Thread object for the algorithm
        """
        if algo_name not in self.algorithms:
            rospy.logerr(f"Algorithm {algo_name} not found")
            return None
        
        # Create thread
        thread = threading.Thread(
            target=self._train_algorithm,
            args=(algo_name,),
            name=f"{algo_name}_thread"
        )
        
        # Store thread reference
        self.threads[algo_name] = thread
        
        # Start thread
        thread.start()
        
        return thread
    
    def run_all(self):
        """Run all algorithms in parallel"""
        # Start visualization if available
        if self.visualization:
            self.visualization.show()
        
        # Start all algorithm threads
        for algo_name in self.algorithms:
            self.run_algorithm(algo_name)
        
        # Wait for all threads to complete
        for algo_name, thread in self.threads.items():
            thread.join()
            rospy.loginfo(f"Thread for {algo_name} completed")
    
    def stop_all(self):
        """Stop all running algorithms (note: this doesn't actually stop them, just waits for completion)"""
        # Wait for all threads to complete
        for algo_name, thread in self.threads.items():
            if thread.is_alive():
                rospy.loginfo(f"Waiting for {algo_name} to complete...")
                thread.join()
    
    def save_results(self):
        """Save results from all algorithms"""
        results_dir = os.path.join(self.config['training']['output_dir'], 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Get results from algorithms that may not have reported yet
        for algo_name, algo in self.algorithms.items():
            if algo_name not in self.results:
                self.results[algo_name] = algo.get_results()
        
        # Save each algorithm's results
        for algo_name, results in self.results.items():
            results_path = os.path.join(results_dir, f"{algo_name}_results.json")
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
            rospy.loginfo(f"Saved {algo_name} results to {results_path}")
        
        # Save combined results summary
        summary = {
            'algorithms': list(self.results.keys()),
            'episodes_trained': self.config['training']['episodes'],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'best_performances': {
                algo: {
                    'best_reward': results.get('best_reward', 0),
                    'training_time': results.get('training_time', 0),
                    'falls': len(results.get('fall_episodes', []))
                }
                for algo, results in self.results.items()
            }
        }
        
        summary_path = os.path.join(results_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        rospy.loginfo(f"Saved summary to {summary_path}")
