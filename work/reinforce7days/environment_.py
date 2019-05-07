from enum import Enum


class State():

    def __init__(self, row=-1, col=-1):
        self.row = row
        self.col = col

    def __repr__(self):
        return "<State: [{}, {}]>".format(self.row, self.col)

    def clone(self):
        return State(self.row, self.col)

    def __hash__(self):
        return hash((self.row, self.col))

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col


class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment():

    def __init__(self, grid, move_prob=0.8):
        self.grid = grid
        self.agent_state = State()
        self.default_reward = -0.04
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        pass

    @property
    def col_length(self):
        pass

    @property
    def actions(self):
        pass

    @property
    def states(self):
        pass

    def transit_func(self, state, action):
        pass

    def can_action_at(self, state):
        pass

    def _move(self, state, action):
        pass

    def reward_func(self, state):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def transit(self, state, action):
        pass
