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
    # plt.violinplot(rewards, showmeans=True)
    sns.violinplot(rewards, palette='pastel', inner='box')
    plt.xticks(range(1, arms+1))
    plt.xlabel('Action')
    plt.ylabel('Reward Distribution')
    plt.show()

def bandit(action, runs):
    return np.random.normal(
        loc=action[np.arange(runs), action],
        scale=1.0
    )

def simple_bandit_algo(arms=10, epsilon=0.1, step_size=1000, runs=2000):
    expected_reward = np.zeros((runs, arms))
    steps = 0
    # for i in range(runs):
    for i in range(step_size):
        if np.random.rand() < epsilon:
            action = np.argmax(
            np.random.random(expected_reward.shape) * (expected_reward==expected_reward.max(axis=1, keepdims=True)), # breaking ties randomly
            axis=1
        )
        else:
            action = np.argmax(
                np.random.random()
            )
        
        action_value = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(runs, arms)
        )

        reward = bandit(action, runs)

        steps += steps + 1
        expected_reward += expected_reward + 1 / steps * (reward - action_value)


    return expected_reward

k_armed_testbed(10)

# simple_bandit_algo()

T = 5
epsilon = 0
for i in range(1, 6):
    for j in range(1, 100):
        epsilon = 1 / 2 * (1 + np.cos(j / 5 * np.pi)) * epsilon
        print(epsilon)

