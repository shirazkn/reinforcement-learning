# Implementation of backward TD model-free prediction of a grid-world
# See Jupyter Notebook for details

import numpy as np
import numpy.random as npr
from tqdm import tqdm
from copy import deepcopy

LAMBDA = 0.5
GAMMA = 1.0


class State:
    value = 0.0
    e_t = 0.0

    def decay_e_t(self):
        self.e_t = GAMMA*LAMBDA*self.e_t

    def increment_e_t(self):
        self.e_t += 1.0

STATES = []
for i in range(9):
    STATES.append([])
    for j in range(9):
        STATES[i].append(State())


class Horse:
    def __init__(self, initial_cell=[0, 0]):
        self.cell = [initial_cell[0], initial_cell[1]]

    def make_move(self):
        dir_1 = npr.randint(0, 4)
        turn = npr.randint(0, 2)

        dir_2 = ((dir_1 - 1) % 4 + 2*turn) % 4

        move(self.cell, dir_1, 1)
        move(self.cell, dir_2, 1)


class Cowboy:
    def __init__(self, initial_cell=[8, 8]):
        self.cell = [initial_cell[0], initial_cell[1]]

    def make_move(self):
        dir = npr.randint(0, 4)
        move(self.cell, dir, 1)


def move(cell, dir, dist):
    # cell is is the starting position
    if dir == 0:
        cell[1] = np.min([8, cell[1] + dist])
    elif dir == 1:
        cell[0] = np.min([8, cell[0] + dist])
    elif dir == 2:
        cell[1] = np.max([-8, cell[1] - dist])
    elif dir == 3:
        cell[0] = np.max([-8, cell[0] - dist])
    # cell is the updated position
    return cell


def chase(n_tries=1, alpha=0.005):
    states = deepcopy(STATES)
    for _ in tqdm(range(n_tries)):
        reset_e_t(states)

        horse = Horse(initial_cell=[0, 0])
        cowboy_house = [npr.randint(0, 9), npr.randint(0, 9)]
        cowboy = Cowboy(initial_cell=cowboy_house)

        chasing = True
        while chasing:
            decay_e_t(states)
            states[cowboy.cell[0]][cowboy.cell[1]].increment_e_t()

            old_cell = deepcopy(cowboy.cell)

            horse.make_move()
            cowboy.make_move()

            if np.all(horse.cell == cowboy.cell):
                reward = 10.0
                chasing = False
            else:
                reward = -1

            td_error = (
                    reward + GAMMA*states[cowboy.cell[0]][cowboy.cell[1]].value
                    - states[old_cell[0]][old_cell[1]].value
            )

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
    X = range(9);
    Y = range(9)
    Z = []
    for i in range(9):
        Z.append([])
        for j in range(9):
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

