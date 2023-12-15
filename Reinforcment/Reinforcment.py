

import matplotlib.pyplot as plt
import gym
import numpy as np
import math
import tracemalloc
tracemalloc.start(25)
# Use TensorFlow v.2 with this old v.1 code.
# E.g. placeholder variables and sessions have changed in TF2.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sess_config = tf.ConfigProto(intra_op_parallelism_threads=1,
                             inter_op_parallelism_threads=1,
                             allow_soft_placement=True)
sess_config.gpu_options.allow_growth=True
sess = tf.Session(config=sess_config)

from reinforcment_learning import *

# TensorFlow
print(tf.__version__)
print(gym.__version__)

#env_name = 'Breakout-v0'
##env_name = 'SpaceInvaders-v0'
#env_name = 'AirRaid-v0'
#env_name = 'Assault-v0'
#env_name = 'DemonAttack-v0'
env_name = 'SpaceInvaders-v0'

checkpoint_base_dir = 'checkpoints_tutorial16/'
update_paths(env_name=env_name)

agent = Agent(env_name=env_name,
                 training=True,
                 render=False,
                 use_logging=True)


model = agent.model
replay_memory = agent.replay_memory
agent.run(num_episodes=1)

#snapshot = tracemalloc.take_snapshot()
#top_stats = snapshot.statistics('traceback')

# pick the biggest memory block
#stat = top_stats[0]
#print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
#for line in stat.traceback.format():
#    print(line)

log_q_values = LogQValues()
log_reward = LogReward()

log_q_values.read()
log_reward.read()

title='Cyclical Epsilon 2500 games'
plt.plot(log_reward.count_states, log_reward.episode, label='Episode Reward')
plt.plot(log_reward.count_states, log_reward.mean, label='Mean of 30 episodes')
plt.xlabel('State-Count for Game Environment')
plt.legend()
#plt.xticks(np.arange(0, int(log_reward.count_states[-1])+1, int(int(log_reward.count_states[-1])/15)))
plt.title(title)
plt.show()

plt.plot(log_q_values.count_states, log_q_values.mean, label='Q-Value Mean')
plt.xlabel('State-Count for Game Environment')
plt.legend()
plt.title(title)
plt.show()

def print_q_values(idx):
    """Print Q-values and actions from the replay-memory at the given index."""

    # Get the Q-values and action from the replay-memory.
    q_values = replay_memory.q_values[idx]
    action = replay_memory.actions[idx]

    print("Action:     Q-Value:")
    print("====================")

    # Print all the actions and their Q-values.
    for i, q_value in enumerate(q_values):
        # Used to display which action was taken.
        if i == action:
            action_taken = "(Action Taken)"
        else:
            action_taken = ""

        # Text-name of the action.
        action_name = agent.get_action_name(i)
            
        print("{0:12}{1:.3f} {2}".format(action_name, q_value,
                                        action_taken))

    # Newline.
    print()

def plot_state(idx, print_q=True):
    """Plot the state in the replay-memory with the given index."""

    # Get the state from the replay-memory.
    state = replay_memory.states[idx]
    
    # Create figure with a grid of sub-plots.
    fig, axes = plt.subplots(1, 2)

    # Plot the image from the game-environment.
    ax = axes.flat[0]
    #ax.set_title('Example State')
    ax.set_title('End Life State')
    ax.imshow(state[:, :, 0], vmin=0, vmax=255,
              interpolation='lanczos', cmap='gray')

    # Plot the motion-trace.
    ax = axes.flat[1]
    ax.set_title('Motion Trace')
    ax.imshow(state[:, :, 1], vmin=0, vmax=255,
              interpolation='lanczos', cmap='gray')

    # This is necessary if we show more than one plot in a single Notebook cell.
    plt.show()
    
    # Print the Q-values.
    if print_q:
        print_q_values(idx=idx)

idx = np.argmax(replay_memory.rewards)
for i in range(-5, 3):
    plot_state(idx=idx+i)

num_used = replay_memory.num_used
q_values = replay_memory.q_values[0:num_used, :]
q_values_min = q_values.min(axis=1)
q_values_max = q_values.max(axis=1)
q_values_dif = q_values_max - q_values_min

#idx = np.argmax(q_values_max)
#for i in range(0, 5):
#    plot_state(idx=idx+i)


idx = np.argmax(replay_memory.end_life)
for i in range(-10, 0):
    plot_state(idx=idx+i)