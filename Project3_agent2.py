import numpy as np
import Project3_agent as ag


class SARSA_0(ag.Agent):
    """implements the SARSA Agent from agent base class"""

    def execute_turn(self, environment):
        """implements the SARSA algorithm for executing a turn"""
        self.turn += 1
        self.state = environment.get_env_state_index()
        self.action = self.select_action(self.state)
        self.next_state, self.reward, game_end = environment.execute_action(self.action)
        self.total_reward += self.reward
        self.select_follow_on_action(self.next_state)
        self.det_q2()
        self.update_q()
        return game_end

    def det_q2(self):
        """determines follow on q value based on on-policy SARSA"""
        self.q2 = self.q[self.next_state][self.next_action]

    def select_action(self, state):
        """uses SARSA algorithm to select the next state"""
        if self.next_action is not None:
            # print("Turn = ", self.turn)
            self.state = state
            # print("State = ", self.state)
            actions = self.q[self.state]
            action = self.e_greedy(actions)
            self.action = action
        else:
            self.action = self.next_action
        return self.action

    def select_follow_on_action(self, state):
        """selects follow-on action, store for action in the next turn"""
        self.next_state = state
        actions = self.q[self.next_state]
        self.next_action = self.e_greedy(actions)
        return self.next_action
