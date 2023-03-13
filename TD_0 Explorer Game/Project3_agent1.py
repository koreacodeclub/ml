import numpy as np
import Project3_agent as ag


class QAgent(ag.Agent):
    """implements the Q Agent"""

    def execute_turn(self, environment):
        """executes one turn in a episode"""
        self.turn += 1
        self.state = environment.get_env_state_index()
        self.action = self.select_action(self.state)
        self.next_state, self.reward, game_end = environment.execute_action(self.action)
        self.total_reward += self.reward
        self.det_q2(self.next_state)
        self.update_q()
        return game_end

    def det_q2(self, next_state):
        """determines the q value based on off policy argmax value"""
        self.q2 = np.amax(self.q[next_state])
        return self.q2
