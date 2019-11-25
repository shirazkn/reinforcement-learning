"""
Implementing a grid maze to study exploration vs. exploitation in tabular learning
Overall code structure borrowed from author of gym_maze(MattChanTK)

REQUIREMENTS :
Install pygame, numpy, gym
Install gym_maze from git@github.com:MattChanTK/gym-maze.git
"""
import gym
import gym_maze

import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from pygame import quit as quit_pygame

losing_streak = 2


def simulate(maze: gym.envs, n_episodes, winning_streak=100, learning_rate=0.01, epsilon=0.3, decay=1.0, policy="EG",
             starting_value=0.0, discount_factor=0.99, display=False):
    """
    :param maze: gym.env
    :param n_episodes: Total # of episodes to attempts, before giving up (must be considerably larger than winning_streak)
    :param winning_streak: <int> # of successes in a row, after which the optimal policy is assumed to have been learnt
    :param learning_rate: <float>
    :param epsilon: Parameter for exploration
    :param decay: Decay rate (for exploration and learning) when policy is eps. decay
    :param policy: "EG", "ED" or "UCB" for epsilon-greedy, decaying epsilon-greedy and UCB policies
    :param starting_value: initialization value for Q(s,a)
    :param discount_factor: Discount for rewards
    :param display: Whether to display the PyGame console
    :return: The episode at which winning_streak was achieved
    """
    maze_size = tuple((maze.observation_space.high + np.ones(maze.observation_space.shape)).astype(int))
    maze_boundary = list(zip(maze.observation_space.low, maze.observation_space.high))

    # If winning_streak is not achieved within this, then terminate
    max_steps = np.prod(maze_size, dtype=int) * 100

    # If maze is solved with more steps than this, then FAIL
    max_steps_for_success = np.prod(maze_size, dtype=int)

    q_table = np.ones(maze_size + (maze.action_space.n,), dtype=float)*starting_value

    # For UCB method, we need to store upper confidence bounds
    if policy == "UCB":
        ucb_table = np.ones(maze_size + (maze.action_space.n,), dtype=float)*epsilon

    success_streak = 0  # Number of times we solved the maze (in a row)
    fail_streak = 0  # Number of times learning failed (in a row)

    # Simulation results
    losses = []
    returns = []
    winning_episode = 0

    if display:
        maze.render()

    for episode in range(n_episodes):

        # Reset the environment
        new_state = maze.reset()

        # the initial state
        state = bound_state(new_state, maze_boundary)
        total_reward = 0

        for t in range(max_steps):

            # Select an action
            # Using EPSILON GREEDY
            if policy == "EG":
                action = select_action_eps_greedy(action_space=maze.action_space, q_values=q_table[state],
                                                  epsilon=epsilon)

            # Using DECAYING-EPSILON GREEDY
            elif policy == "ED":
                action = select_action_eps_greedy(action_space=maze.action_space, q_values=q_table[state],
                                                  epsilon=epsilon)
                epsilon = np.max([epsilon*decay, 0.001])
                learning_rate = np.max([learning_rate*decay, 0.001])

            # Using UPPER CONFIDENCE BOUNDS
            elif policy == "UCB":
                action = select_action_ucb(q_values=q_table[state], ucbs=ucb_table[state])
                ucb_table[state][action] *= decay

            # Execute the action
            new_state, reward, solved, _ = maze.step(action)

            # Observe the reward
            new_state = bound_state(new_state, maze_boundary)
            total_reward += reward

            # Update Q(s,a)
            q_max = np.amax(q_table[new_state])
            loss = reward + discount_factor * q_max - q_table[state + (action,)]
            q_table[state + (action,)] += learning_rate * loss

            # For next iteration
            state = new_state
            losses.append(loss)

            # Render PyGame frame
            if display:
                maze.render()

            # Update # of fails in a row
            if t == max_steps - 1:
                fail_streak += 1

            if solved:
                fail_streak = 0
                returns.append(total_reward)

                # Update # of successes in a row
                if t <= max_steps_for_success:
                    success_streak += 1
                else:
                    success_streak = 0
                break

        # Conditions for Win / Loss
        # If <losing_streak> # of failures were achieved in a row
        if fail_streak > losing_streak:
            # print(f"Failed {losing_streak} times in a row...")
            break

        # If <winning_streak> # of successes were achieved in a row
        if success_streak > winning_streak:
            winning_episode = episode
            break

    try:
        return {
            "winning_episode": winning_episode,
            "losses": losses,
            "avg_losses": [np.average(losses[i*10:(i+1)*10]) for i in range(int(len(losses)/10))],
        }

    finally:
        quit_pygame()


def select_action_eps_greedy(action_space, q_values, epsilon):
    """
    :return: Epsilon-greedy action
    """
    # Select a random action
    if random.random() < epsilon:
        action = action_space.sample()
    # Select greedy action
    else:
        action = int(np.argmax(q_values))
    return action


def select_action_ucb(q_values, ucbs):
    """
    :param q_values: Q values for each action at current state
    :param ucbs: Upper confidence bounds for each Q value
    :return: Action based on 'UCB' exploration method
    """
    action = int(np.argmax(np.array(q_values) + np.array(ucbs)))
    return action


def bound_state(state, bounds):
    """
    Bounds each element in `state` within the limits specified in `bounds`
    """
    indices = []
    for i in range(len(state)):
        index = int(max(state[i], bounds[i][0]))
        index = int(min(index, bounds[i][1]))
        indices.append(index)

    return tuple(indices)


def average_win_episode(params, n_episodes, n_mazes=100):
    """
    Averages over n_simulations different mazes
    :param params: <dict> kwargs for simulate()
    :param n_episodes: n_episodes for simulate()
    :param n_mazes: Number of mazes to average over
    :return: Average number of episodes before winning streak was achiever
    """
    win_episodes = []
    fail_streak = 0
    for i in range(n_mazes):
        win_episode = simulate(new_maze(), n_episodes, **params)["winning_episode"]

        # Winning episode is 0 if learning was failed
        if win_episode:
            fail_streak = 0
            win_episodes.append(win_episode)

        else:
            fail_streak += 1

        # Two fails in a row >> Terminate
        if fail_streak > 2:
            break

    return np.average(win_episodes) if (len(win_episodes)) else 0


def plot_episodes_against_parameter(param_name: str, param_values, other_params, n_mazes=10, view_plot=True):
    """

    :param param_name: Name of parameter to plot against
    :param param_values: Values for parameter
    :param other_params: Other parameters for simulate()
    :param n_mazes: Number of mazes to average over
    :param view_plot: Can suppress plot output (if you just want return value)
    :return: <list> of length len(param_values)
    """
    params = other_params.copy()
    n_episodes = 200
    win_episodes = []

    for val in tqdm.tqdm(param_values):
        # Get win_episode for a new maze
        params[param_name] = val
        win_episode = average_win_episode(params, n_episodes=n_episodes, n_mazes=n_mazes)

        # If learning failed...
        if not win_episode:
            win_episodes.append(n_episodes)

        # Else success
        else:
            win_episodes.append(win_episode)

    if view_plot:
        min_episodes = np.min(win_episodes)
        if min_episodes < n_episodes:
            print(f"Policy was learnt as quickly as {min_episodes}.")

        # Plot episodes against param_values
        plt.plot(param_values, win_episodes)
        plt.xlabel(param_name.capitalize())
        plt.ylabel("Number of Episodes")
        plt.show()

    return win_episodes


def new_maze(display=False):
    """
    Make a randomized maze
    :param display: Whether to render PyGame frames
    :return: gym.env-like object
    """
    return gym.make("maze-random-10x10-plus-v0", enable_render=display)
