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

def simple_bandit(k, epsilon, steps, initial_Q, alpha=0):
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