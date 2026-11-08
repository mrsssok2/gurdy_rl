#!/usr/bin/env python3
"""
Greedy (epsilon 0) action selection with a saved DQN model, shared by
evaluate.py and voice_command_node.py.

dqn_features() must match extract_state_features in train_gurdy_dqn.py,
which can't be imported outside training because it opens the training
plots on import.
"""
import os

import numpy as np

import env as gurdy_env

PACKAGE_NAME = "my_gurdy_description"


def dqn_features(observation):
    """DQN input for an env observation (see train_gurdy_dqn.extract_state_features)."""
    obs = np.asarray(observation, dtype=np.float32)
    return np.concatenate([
        obs[gurdy_env.OBS_LEG_CODES],
        [obs[gurdy_env.OBS_FORWARD_VELOCITY] / 2.0],
        [obs[gurdy_env.OBS_FORWARD_DISTANCE] / 10.0],
        [obs[gurdy_env.OBS_HEAD_HEIGHT] / 0.3],
        obs[gurdy_env.OBS_LAST_ACTION.start:gurdy_env.OBS_SIZE],
    ]).astype(np.float32)


def greedy_action(model, observation):
    """Action with the highest Q value, as DQN.act does with epsilon 0."""
    state = dqn_features(observation).reshape(1, -1)
    q_values = model.predict(state, verbose=0)[0]
    return int(np.argmax(q_values))


def resolve_model_path(path):
    """Absolute path; a relative one is taken relative to the package."""
    if path and not os.path.isabs(path):
        import rospkg
        path = os.path.join(rospkg.RosPack().get_path(PACKAGE_NAME), path)
    return path


def load_model(path):
    """Load a saved Keras DQN model (gurdy_dqn_model_<episode>)."""
    from tensorflow import keras
    return keras.models.load_model(path)
