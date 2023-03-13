import Project3_env as env
import Project3_agent1 as agq
import Project3_agent2 as ags
import matplotlib.pyplot as plt


def main():
    """this is the main application"""
    app = Application()
    app.welcome()
    app.run()


class Application:

    """class to deal with the mechanics of the application"""

    def __init__(self):
        self.environment = env.ExplorerGame()
        self.q = agq.QAgent()
        self.sarsa = ags.SARSA_0()
        self.welcome_msg = "Welcome to the Explorer Game 0.1"
        self.author = "Josh Gompert"
        self.experiment = None
        self.agent = None
        self.iterations = None
        self.end_game = False
        self.exp_starts = False
        self.terminate = False
        self.decrease_epsilon = False
        self.hyper_modded = True  # must be modified in code used for graphing purposes only

    def welcome(self):
        print(self.welcome_msg)
        print("Developed by: %s" % self.author)

    def get_input(self):
        """gets user input to run the program or experiment"""
        action = int(input("Select an action: 1 = learn, 2 = print q table, 3 = conduct experiment 9 = exit"))
        if action == 1:
            agent_selection = input("Select an Agent: q or s")
            if agent_selection.lower() == "q":
                self.agent = self.q
            else:
                self.agent = self.sarsa
            self.iterations = int(input("Select the number of times to run: <int>"))
            self.learn()
        elif action == 2:
            print(self.agent.q)
            print("percent not visited = %s" % self.agent.det_not_visited())
        elif action == 3:
            self.experiment = None
            condition = int(input("Select a experiment condition: <int> (1 - 4)"))
            dec_e = input("Do you want to decrement epsilon? y or n")
            if dec_e.lower() == "y":
                self.decrease_epsilon = True
            else:
                self.decrease_epsilon = False
            self.experiment = Experiment(self, condition)
            self.experiment.run_experiment()
        elif action == 9:
            self.terminate = True

    def episode(self):
        """executes a single episode of the explorer game"""
        self.end_game = False
        while not self.end_game:
            self.end_game = self.agent.execute_turn(self.environment)
        print("total reward = %s" % self.agent.total_reward)

    def learn(self, experiment=False):
        """continues to execute the explorer game for a pre-determined number of iterations"""
        rewards = []
        visited = []
        epsilons = []
        print("Learning using agent %s for %s iterations." % (self.agent, self.iterations))
        for i in range(0, self.iterations):
            self.reset()
            self.episode()
            print("completed iteration # %s" % i)
            if experiment:
                rewards.append(self.agent.total_reward)
                visited.append(self.agent.det_not_visited())
                if self.decrease_epsilon:
                    epsilons.append(self.agent.epsilon)
                    self.agent.reduce_epsilon(self.iterations)
        return rewards, visited, epsilons

    def reset(self):
        self.environment.reset(self.exp_starts)
        self.agent.reset()

    def run(self):
        while not self.terminate:
            self.end_game = False
            self.get_input()


def average_results(dataset):
    """averages the results from the k agents"""
    averages = []
    n = len(dataset[0])  # number of iterations
    k = len(dataset)  # number of agents
    for i in range(0, n):
        total = 0
        for j in range(0, k):
            total += dataset[j][i]
        averages.append(total / k)
    return averages


class Experiment:

    """class to run the experiments as required by project 3"""

    def __init__(self, application, condition, number_of_agents=10, num_of_iterations=1000):
        self.application = application
        self.application.iterations = num_of_iterations
        self.condition = condition
        if self.condition == 1:
            self.zeros = True
            self.application.exp_starts = False
        elif self.condition == 2:
            self.zeros = True
            self.application.exp_starts = True
        elif self.condition == 3:
            self.zeros = False
            self.application.exp_starts = False
        else:
            self.zeros = False
            self.application.exp_starts = True
        self.title = "Experiment " + str(condition)
        self.q_agents = []
        self.s_agents = []
        self.q_totals = []
        self.s_totals = []
        self.q_p_visited = []
        self.s_p_visited = []
        self.epsilons = []
        self.epsilons_documented = False
        for i in range(0, number_of_agents):
            self.q_agents.append(agq.QAgent(zero=self.zeros))
            self.s_agents.append(ags.SARSA_0(zero=self.zeros))
            print("added agents %s" % i)

    def run_experiment(self):
        for q in self.q_agents:
            self.application.agent = None
            self.application.agent = q
            rewards, visited, epsilons = self.application.learn(experiment=True)
            self.q_totals.append(rewards)
            self.q_p_visited.append(visited)
            if self.application.decrease_epsilon:
                if not self.epsilons_documented:
                    self.epsilons = epsilons
                    print(self.epsilons)
                    self.epsilons_documented = True
        for s in self.s_agents:
            self.application.agent = None
            self.application.agent = s
            rewards, visited, epsilons = self.application.learn(experiment=True)
            self.s_totals.append(rewards)
            self.s_p_visited.append(visited)
        self.q_totals = average_results(self.q_totals)
        self.q_p_visited = average_results(self.q_p_visited)
        self.s_totals = average_results(self.s_totals)
        self.s_p_visited = average_results(self.s_p_visited)
        self.produce_graph()

    def produce_graph(self):
        x = []
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        self.title = self.title + "with Coins"
        if self.epsilons_documented:
            self.title = self.title + " Epsilon Decreased"
        if self.application.hyper_modded:
            self.title = self.title + "\n learning_rate = %s; discount = %s" % (self.application.agent.learning,
                                                                                self.application.agent.y)
        plt.title(self.title)

        for i in range(0, self.application.iterations):
            x.append(i)

        fig.suptitle(self.title)

        axs[0].plot(x, self.q_totals, color='blue', label="Q-Learning")
        axs[0].plot(x, self.s_totals, color='red', label="SARSA")
        axs[0].set_xlabel("# of iterations")
        axs[0].set_ylabel("average total reward")
        axs[0].set_title("Reward Over Iterations")
        axs[0].legend()

        if self.epsilons_documented:
            axs[1].plot(x, self.epsilons, color='black')
            axs[1].set_title("Epsilon Over Time")
            axs[1].set_xlabel("# of iterations")
            axs[1].set_ylabel("epsilon")
        else:
            axs[1].plot(x, self.q_p_visited, color='blue')
            axs[1].plot(x, self.s_p_visited, color='red')
            axs[1].set_title("State-Action Pairs Visited")
            axs[1].set_xlabel("# of iterations")
            axs[1].set_ylabel("percent visited")

        plt.show()


if __name__ == "__main__":
    main()
