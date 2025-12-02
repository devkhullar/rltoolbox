'''A replication of the Simple Bandit Problem.'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

def k_armed_testbed(arms:int=10, runs:int=2000, iteration:int=0):
    '''A testbed for the k-armed bandit problem
    
    Parameters
    ----------
    arms : int
        Number of actions.
    runs : int
        Number of independent runs that the bandit problem takes.
    iteration:
        Take the sample of the iteration to generate the violinplot.

    Returns
    -------
    Violin plot 
        '''
    action_values = np.random.normal(loc=0.0, scale=1.0, size=(runs, arms))
    rewards = [0] * arms
    for i in range(arms):
        rewards[i] = np.random.normal(action_values[iteration, i], scale=1.0, size=runs)
    # plt.violinplot(rewards, showmeans=True)
    sns.violinplot(rewards, palette='pastel', inner='box')
    plt.xticks(range(1, arms+1))
    plt.xlabel('Action')
    plt.ylabel('Reward Distribution')
    plt.show()

def bandit(action, problem, action_value):
    return np.random.normal(q_star[problem, action], 1)

def simple_max(Q):
    return np.random.choice(np.flatnonzero(Q == Q.max())) # breaking ties randomly

def simple_bandit(k, epsilon, num_problems, q_star, steps, initial_Q, alpha=0):
    rewards = np.zeros(steps)
    actions = np.zeros(steps)
    
    for i in tqdm(range(num_problems)):
        Q = np.ones(k) * initial_Q # initial Q
        N = np.zeros(k)  # initalize number of rewards given
        best_action = np.argmax(q_star[i])
        for t in range(steps):
            if np.random.rand() < epsilon: # explore
                a = np.random.randint(k)
            else: # exploit
                a = simple_max(Q, N, t)

            reward = bandit(a, i)

            N[a] += 1
            if alpha > 0:
                Q[a] = Q[a] + (reward - Q[a]) * alpha
            else:
                Q[a] = Q[a] + (reward - Q[a]) / N[a]

            rewards[t] += reward
            
            if a == best_action:
                actions[t] += 1
    
    return np.divide(rewards,num_problems), np.divide(actions,num_problems)

# https://www.spktsagar.com/posts/2023/06/rl-multiarmed-bandit/

class StationaryMultiArmedBandit:
    """
    Represents a stationary multi-armed bandit problem.

    Attributes:
        k (int): Number of arms.
        runs (int): Number of independent runs.
        random_state (int, optional): Random seed for reproducibility.
    """
    def __init__(
            self,
            k,
            runs,
            random_state=None,
    ):
        self.k = k
        self.runs = runs
        self.random_state = random_state

        self.setup()

    def setup(self):
        """Set up the seed for reproducibility and reward distribution"""
        self.nprandom = np.random.RandomState(self.random_state)
        self.q_star = self.nprandom.normal(
            loc=0.0,
            scale=1.0,
            size=(self.runs, self.k),
        )

    def get_reward(self, action):
        """Given the action, return the reward"""
        reward = self.nprandom.normal(
            loc=self.q_star[np.arange(self.runs), action],
            scale=1.0,
        )
        return reward

    def get_correct_action(self):
        """
        Get the correct action for each run.
        Correct action for each run is the one with highest mean reward
        """
        return self.q_star.argmax(axis=1)

    def plot_reward_distribution(self, run=0):
        """Plot the reward distribution for the given run."""
        samples = self.nprandom.normal(
            loc=self.q_star[run],
            scale=1.0,
            size=(10_000, self.k),
        )
        plt.violinplot(samples, showmeans=True)
        plt.xlabel('Action')
        plt.ylabel('Reward Distribution')
        plt.show()

class Agent:
    """
    An epsilon-greedy agent using sample-average method for action value estimation.

    Attributes
    ----------
    k : int
        Number of actions.
    runs : int
        Number of independent runs.
    epsilon : float, optional
        Probability of choosing a random action (exploration), default is 0.1.
    random_state : int, optional
        The random number generator seed to be used, default is None.
    """
    def __init__(
            self,
            k,
            runs,
            epsilon=0.1,
            random_state=None,
    ):
        self.k = k
        self.runs = runs
        self.epsilon = epsilon
        self.random_state = random_state

        self.setup()

    def setup(self):
        """Initialize the Q and N arrays for action value estimation and action counts."""
        self.nprandom = np.random.RandomState(self.random_state)
        self.Q = np.zeros((self.runs, self.k))
        self.N = np.zeros((self.runs, self.k))

    def get_action(self):
        """Choose an action based on epsilon-greedy policy."""
        greedy_action = np.argmax(
            self.nprandom.random(self.Q.shape) * (self.Q==self.Q.max(axis=1, keepdims=True)), # breaking ties randomly
            axis=1
        )
        random_action = self.nprandom.randint(0, self.k, size=(self.runs, ))

        action = np.where(
            self.nprandom.random((self.runs, )) < self.epsilon,
            random_action,
            greedy_action,
        )
        return action

    def get_step_size(self, action):
        """Calculate the step size for updating action value estimates.
        For sample average method we return 1/number of times the action is choosen until current step"""
        return 1/self.N[np.arange(self.runs), action]

    def update(self, action, reward):
        """Update the action value estimates based on the chosen action and received reward."""
        self.N[np.arange(self.runs), action] += 1
        step_size = self.get_step_size(action)
        self.Q[np.arange(self.runs), action] += (reward - self.Q[np.arange(self.runs), action])*step_size

class Bandit:
    """A test bed for running experiments with multi-armed bandits and agents.

    Attributes:
        bandit (object): A multi-armed bandit object.
        agent (object): An agent object.
        steps (int): The number of steps for the experiment.
    """
    def __init__(
            self,
            bandit,
            agent,
            steps,
    ):
        self.bandit = bandit
        self.agent = agent
        self.steps = steps

    def run_experiment(self):
        """Runs the experiment for the given number of steps and returns the average rewards and optimal actions.

        Returns:
            tuple: A tuple containing two lists: average rewards and average optimal actions for each step.
        """
        avg_reward = []
        avg_optimal_action = []

        for _ in range(self.steps):
            action = self.agent.get_action()
            reward = self.bandit.get_reward(action)
            self.agent.update(action, reward)

            correct = action == self.bandit.get_correct_action()

            avg_reward.append(reward.mean())
            avg_optimal_action.append(correct.mean())

        return avg_reward, avg_optimal_action

    @classmethod
    def run_and_plot_experiments(cls, steps, exp_bandit_agent_dict):
        """Runs multiple experiments and plots the results.

        Args:
            steps (int): The number of steps for the experiments.
            exp_bandit_agent_dict (dict): A dictionary with labels as keys and (bandit, agent) tuples as values.
        """
        # fig, (ax_reward, ax_optimal_action) = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

        for label, (bandit, agent) in exp_bandit_agent_dict.items():
            test_bed = cls(bandit, agent, steps)
            avg_reward, avg_optimal_action = test_bed.run_experiment()
            # ax_reward.plot(avg_reward, label=label)
            # ax_optimal_action.plot(avg_optimal_action, label=label)

        return avg_reward, avg_optimal_action

        # ax_reward.set_ylabel("Average reward")
        # ax_reward.set_xlabel("Steps")

        # ax_optimal_action.set_ylabel("% Optimal Action")
        # ax_optimal_action.set_xlabel("Steps")

        # ax_reward.legend()
        # ax_optimal_action.legend()

        # plt.show()

def epsilon_experiment(T, alpha, epsilon_0, logscale=True):
    epsilon_list = []
    for i in range(T):
        epsilon = 1 / 2 * (1 + np.cos(i * np.pi / alpha)) * epsilon_0
        epsilon_list.append(epsilon)

    plt.plot(epsilon_list)
    plt.xlabel('Epoch')
    plt.ylabel('Epsilon')
    if logscale: plt.yscale('log')
    plt.show()

def simple_bandit_problem(arms, runs, steps, epsilon):
    reward, action = Bandit.run_and_plot_experiments(
        steps=steps,
        exp_bandit_agent_dict={
            rf'$\epsilon={epsilon}$' : (
                StationaryMultiArmedBandit(
                    k=arms,
                    runs=runs
                    ), 
                Agent(
                    k=arms,
                    runs=runs,
                    epsilon=epsilon
                    )
                )
            }
        )

    return reward, action

def reward_vs_epsilon(T0, total_iterations, epsilon_0, arms, runs, steps, plot=True):
    '''
    Test how the distribution of reward changes with a change of 
    the epsilon parameter.

    Parameters
    ----------
    k. : Number of epochs
    T0 : Sub-epoch; one of the intervals of the epoch, for which the
        problem is tested.
    '''
    rewards_list = []
    epsilon_list = []
    k = 0
    for i in range(1):
        while k < total_iterations:
            j = k % T0
            epsilon = 1 / 2 * (1 + np.cos(j * np.pi / T0)) * epsilon_0
            epsilon_list.append(epsilon)
            reward, action = simple_bandit_problem(
                arms=arms,
                runs=runs,
                steps=steps, 
                epsilon=epsilon
            )
            rewards_list.append(np.mean(reward))
            k += 1
            print(f'\r{k} / {total_iterations} iterations done', end='', flush=True)
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(epsilon_list, rewards_list)
        ax.set_xlabel(r'$\epsilon$')
        ax.set_ylabel('Average Reward')
        ax.set_title(rf'$\epsilon_0$ = {epsilon_0}')
        # plt.show()

    return rewards_list, epsilon_list
    
