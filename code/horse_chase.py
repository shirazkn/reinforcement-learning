# Implementation of backward TD model-free prediction of a grid-world
# See Jupyter Notebook for details

import numpy as np
import numpy.random as npr
from tqdm import tqdm
from copy import deepcopy

# Parameter for TD(Lambda)
LAMBDA = 0.5

# Discount factor = 1 (un-discounted rewards)
GAMMA = 1.0

# Four directions for traversing the grid
DIRECTIONS = ["Up", "Right", "Down", "Left"]


class State:
    # Initial guess for state value
    value = 0.0

    # Eligibility trace of each state starts as 0.0
    e_t = 0.0

    # Decay eligibility trace at every time-step
    def decay_e_t(self):
        self.e_t = GAMMA*LAMBDA*self.e_t

    # Bump eligibility trace by 1; Called when this state is visited
    def increment_e_t(self):
        self.e_t += 1.0


# 9x9 states, for 9x9 positions of Cowboy
STATES = []
for i in range(9):
    STATES.append([])
    for j in range(9):
        STATES[i].append(State())


class Horse:
    def __init__(self, initial_cell=None):
        initial_cell = [0, 0] if initial_cell is None else initial_cell
        self.cell = [initial_cell[0], initial_cell[1]]

    def make_move(self):
        # Pick a direction, out of [Up, Right, Down, Left]
        direc_1 = npr.randint(0, 4)
        # Move one step
        move(self.cell, direc_1, 1)

        # Turn either left or right
        turn = npr.randint(0, 2)
        direc_2 = ((direc_1 - 1) % 4 + 2*turn) % 4
        # Move one step
        move(self.cell, direc_2, 1)


class Cowboy:
    def __init__(self, initial_cell):
        self.cell = [initial_cell[0], initial_cell[1]]

    def make_move(self):
        # Pick a direction, out of [Up, Right, Down, Left]
        direc = npr.randint(0, 4)
        # Move one step
        move(self.cell, direc, 1)


def move(cell, direc, dist):
    # Start at cell and move dist units in a direction

    # cell is the starting position
    if direc == 0:
        cell[1] = np.min([8, cell[1] + dist])
    elif direc == 1:
        cell[0] = np.min([8, cell[0] + dist])
    elif direc == 2:
        cell[1] = np.max([-8, cell[1] - dist])
    elif direc == 3:
        cell[0] = np.max([-8, cell[0] - dist])
    # cell is now the updated position
    return cell


def TD_Lambda(n_tries=1, alpha=0.005):

    states = deepcopy(STATES)
    # Since we are copying a 2-D array, copy() doesn't suffice

    for _ in tqdm(range(n_tries)):
        # Set all eligibility traces to 0.0
        reset_e_t(states)

        # Place horse at (0,0)
        horse = Horse(initial_cell=[0, 0])

        # Cowboy starts at a random cell
        cowboy_initial = [npr.randint(0, 9), npr.randint(0, 9)]
        cowboy = Cowboy(initial_cell=cowboy_initial)

        chasing = True
        while chasing:
            # All eligibility traces decrease every time-step
            decay_e_t(states)

            # E_t of the current state increases by 1.0
            states[cowboy.cell[0]][cowboy.cell[1]].increment_e_t()

            # State S_t, in Silver's lecture slides
            old_cell = deepcopy(cowboy.cell)

            # Get state S_t+1
            cowboy.make_move()
            horse.make_move()

            # Assign reward for cowboy
            if np.all(horse.cell == cowboy.cell):
                reward = 10.0
                chasing = False
            else:
                reward = -1

            # TD-error for time-step t+1
            td_error = (
                    reward + GAMMA*states[cowboy.cell[0]][cowboy.cell[1]].value
                    - states[old_cell[0]][old_cell[1]].value
            )

            # Update all states with TD-error, weighted by their eligibility trace and step-size
            td_update_all(states, td_error, alpha=alpha)

    return states


def td_update_all(states, error, alpha):
    for i in range(9):
        for j in range(9):
            states[i][j].value += alpha*error*states[i][j].e_t


def decay_e_t(states):
    for i in range(9):
        for j in range(9):
            states[i][j].decay_e_t()


def reset_e_t(states):
    for i in range(9):
        for j in range(9):
            states[i][j].e_t = 0.0


from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm

def plot(states):
    fig = plt.figure();
    ax = fig.gca(projection='3d')
    X = []
    Y = []
    Z = []
    for i in range(9):
        X.append([])
        Y.append([])
        Z.append([])
        for j in range(9):
            X[-1].append(i)
            Y[-1].append(j)
            Z[-1].append(states[i][j].value)

    surf = ax.plot_surface(X, Y, np.array(Z), cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def c_plot(states):
    X = range(9)
    Y = range(9)
    Z = []
    for i in range(9):
        Z.append([])
        for j in range(9):
            Z[-1].append(states[i][j].value)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    plt.imshow(Z, extent=(X.min(), X.max(), Y.max(), Y.min()),

               interpolation='nearest', cmap=cm.gist_rainbow)

    plt.show()

