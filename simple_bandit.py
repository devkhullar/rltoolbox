'''A replication of the Simple Bandit Problem.'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    print(rewards)
    # plt.violinplot(rewards, showmeans=True)
    sns.violinplot(rewards, palette='pastel', inner='box')
    plt.xticks([i for i in range(1, arms+1)])
    plt.xlabel('Action')
    plt.ylabel('Reward Distribution')
    plt.show()


def simple_bandit_algo(arms=10, epsilon=0.1, step_size=1000, runs=2000):
    expected_reward = np.zeros(arms)
    steps = 0
    # for i in range(runs):
    for j in range(steps):
        if np.random.rand() < epsilon:
            action = np.random.randint(arms)
        else:
            action = np.argmax(expected_reward)







