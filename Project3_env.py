import random


def get_state_index(x, y):
    x_idx = x - 1
    y_idx = 8 * (y - 1)
    return x_idx + y_idx  # ranges from 0 to 63


class ExplorerGame:
    """The exploring for gold puzzle"""

    def __init__(self):

        self.num_states = 64
        self.num_actions = 4
        self.expl_x = 1  # explorer's x position from 1 to 8
        self.expl_y = 1  # explorer's y position from 1 to 8
        self.win = {15, 62}
        self.loss = {13, 17, 28, 32, 53}
        self.coins = {20, 50}
        self.mount = {1, 9, 49}
        self.state = None  # added to store the state

        # Get the key environment parameters
    def get_number_of_states(self):
        return self.num_states

    def get_number_of_actions(self):
        return self.num_actions

    # Get the state IDs that should not be set optimistically
    def get_terminal_states(self):
        term = self.win.union(self.loss, self.mount)
        return term

    def get_state(self):
        return get_state_index(self.expl_x, self.expl_y)

        # Set the current state to the initial state
    def reset(self, exp_starts):
        x = 1
        y = 1
        if exp_starts:
            done = False
            while not done:
                x = random.randint(1, 8)
                y = random.randint(1, 8)
                st = get_state_index(x, y)
                if (st in self.win) or (st in self.loss) or (st in self.mount):
                    done = False
                else:
                    done = True
        self.expl_x = x
        self.expl_y = y
        self.state = get_state_index(self.expl_x, self.expl_y)
        return self.state

    def execute_action(self, action):
        # Use the agent's action to determine the next state and reward #
        # Note: 'up' = 0, 'down' = 1, 'left' = 2, 'right' = 3 #

        current_state = get_state_index(self.expl_x, self.expl_y)
        new_state = current_state
        reward = 0
        game_end = False
        # if in terminal states, stay in terminal states
        if (current_state in self.win) or (current_state in self.loss):
            game_end = True
        # if in undefined states, stay in undefined states
        elif current_state in self.mount:
            game_end = False
        else:
            # determine a potential next state
            temp_x = self.expl_x
            temp_y = self.expl_y
            if action == 0:
                if temp_y > 1:      # action is up
                    temp_y = temp_y - 1
            elif action == 1:
                if temp_y < 8:    # action is down
                    temp_y = temp_y + 1
            elif action == 2:
                if temp_x > 1:    # action is left
                    temp_x = temp_x - 1
            else:                 # action is right
                if temp_x < 8:        # print(self.next_state, self.reward)
                    temp_x = temp_x + 1

            new_state = get_state_index(temp_x, temp_y)
            if new_state in self.mount:
                new_state = current_state
                reward = -1
                game_end = False
            else:
                self.expl_x = temp_x
                self.expl_y = temp_y
                if new_state in self.coins:
                    reward = 2  # modified for coins
                    game_end = False
                elif new_state in self.loss:
                    reward = -20
                    game_end = True
                elif new_state in self.win:
                    reward = 20
                    game_end = True
                else:
                    reward = -1
                    game_end = False
        # print("new_state =", new_state, "reward = ", reward, "game_end =", game_end)
        if game_end:
            print(" ")
        return new_state, reward, game_end

    def get_env_state_index(self):
        """added to provide a state value to the environment"""
        x_idx = self.expl_x - 1
        y_idx = 8 * (self.expl_y - 1)
        self.state = x_idx + y_idx
        return x_idx + y_idx  # ranges from 0 to 63



