import numpy as np


class Agent:
    """base class to learn using RL algorithms"""

    def __init__(self, zero=True):
        # to be implemented
        self.num_states = 64
        self.num_actions = 4
        self.s_a_pairs = self.num_actions * self.num_states
        if zero:
            self.q = np.zeros((64, 4), dtype="float64")
        else:
            self.q = np.full((64, 4), 25, dtype="float64")
        self.zero = zero
        self.state = None
        self.next_state = None
        self.action = None
        self.next_action = None
        self.q1 = None
        self.q2 = None
        self.reward = None
        self.total_reward = 0
        self.turn = 0
        self.epsilon = 0.2
        self.learning = .01
        self.y = 0.995
        self.visited = 0
        self.percent_not_visited = 1

    def get_number_of_states(self):
        """returns the number of states"""
        return self.num_states

    def get_number_of_actions(self):
        """returns the number of actions """
        return self.num_actions

    def e_greedy(self, actions):
        """implement epsilon greedy selection"""
        a_star_idx = np.argmax(actions)
        rng = np.random.default_rng()
        if self.epsilon <= rng.random():
            return a_star_idx
        else:
            b = actions.size
            idx = rng.integers(low=0, high=b)
            return idx

    def select_action(self, state):
        """selects an action based on the state using epsilon greedy"""
        # print("Turn = ", self.turn)
        self.state = state
        # print("State = ", self.state)
        actions = self.q[state]
        action = self.e_greedy(actions)
        self.action = action
        return self.action

    def execute_turn(self, environment):
        """executes one turn in a episode"""
        return False

    def update_q(self):
        """updates the q table"""
        self.q1 = self.q[self.state][self.action]
        self.q[self.state][self.action] = self.q1 + self.learning * (self.reward + (self.y * self.q2) - self.q1)

    def reset(self):
        """resets the trajectory to a blank list"""
        self.state = None
        self.next_state = None
        self.action = None
        self.next_action = None
        self.q1 = None
        self.q2 = None
        self.reward = None
        self.total_reward = 0
        self.turn = 0

    def det_not_visited(self):
        """determines the percent of state action pairs not visited, returns percentage"""
        if self.zero:
            self.visited = np.count_nonzero(self.q)
        else:
            self.visited = np.count_nonzero(self.q != 25)
        print("visited %s state-action pairs" % self.visited)
        self.percent_not_visited = (self.visited / self.s_a_pairs) * 100
        return self.percent_not_visited

    def reduce_epsilon(self, total_iterations):
        """reduces the epsilon during the learning process"""
        self.epsilon = self.epsilon - (self.epsilon * (10 / total_iterations))



