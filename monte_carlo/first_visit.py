import gym
import numpy as np 
from tqdm import tqdm

class FirstVisitMC():
    """
    Implementation of the on-policy first-visit Monte Carlo method for epsilon-soft policies.
    """

    def __init__(self, env: gym.Env, gamma: float = .9, eps: float = .1):
        self.env = env
        self.gamma = gamma
        self.eps = eps
        self.ns = self.env.observation_space.n
        self.na = self.env.action_space.n
        self.policy = self.initialise_policy()
        self.Q = self.initialise_value_function()
        self.returns = self.initialise_returns()

    def initialise_policy(self):
        """
        Initialise the policy to a random policy.

        A policy is a matrix of size (na, ns), where na is the number of actions and ns is the number of states.
        The columns of the matrix (state) are the probability distributions over actions given that state pi(a|s).
        """
        policy = np.random.uniform(size=(self.na, self.ns))
        policy = np.divide(policy, policy.sum(axis=0))
        return policy

    def sample_action(self, state: int):
        """
        Sample an action according to the current policy.
        
        Parameters
        ----------
        state : int
            The state to sample an action from. 
            It is an integer between 0 and self.ns-1.
        """
        dist = self.policy[:,state]
        idx = np.random.multinomial(1, dist, size=None)
        return np.where(idx == 1)[0][0]

    def initialise_value_function(self):
        """
        Initialise the value function to zero.
        """
        #Q = np.zeros((self.na, self.ns))
        Q = np.random.uniform(
            low=0,
            high=1,
            size=(self.na, self.ns)
            )
        return Q

    def initialise_returns(self):
        """
        Initialise the returns dictionary.
        """
        returns = {}

        for state in range(self.ns):
            for action in range(self.na):
                returns[(action, state)] = []
        return returns
    
    def generate_episode(self, render=False, verbose=False):
        """
        Generate an episode using the current policy.
        """
        if verbose:
            print("--------NEW EPISODE--------")
        episode = []
        state = self.env.reset()
        done = False
        step = 0
        while not done:
            
            if render:
                self.env.render()

            action = self.sample_action(state)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            
            if verbose:
                print(f"Step: {step}")
                print(f"State: {state}")
                print(f"Action: {action}")
                print(f"Reward: {reward}")
                print(f"Next state: {next_state}")
                print(f"Done: {done}")
                print("\n")

            state = next_state
            step += 1

        self.env.close()
        return episode


    def train(self, n_episodes: int = 1000, verbose=False):
        """
        Train the policy using the first-visit Monte Carlo method.
        """
        for episode in tqdm(range(n_episodes)):
            episode = self.generate_episode(verbose=verbose)
            G = 0

            while episode:
                state, action, reward = episode.pop()
                G = self.gamma * G + reward

                if (state, action) not in {(s,a) for s,a,_ in episode}:
                    self.returns[(action, state)].append(G)
                    self.Q[action, state] = np.mean(self.returns[(action, state)])
                    a_greedy = np.argmax(self.Q[:,state])
                    self.policy[:,state] = self.eps / (self.na - 1)
                    self.policy[a_greedy, state] = 1 - self.eps


if __name__=='__main__':
    import pickle
    import os
    from utils import plot_valuefunc_and_policy

    # create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # Train the agent
    env = gym.make('FrozenLake8x8-v1', is_slippery=False)
    mc = FirstVisitMC(env, gamma=.99, eps=.3)
    mc.train(n_episodes=10000)

    # Plot the value function and policy
    plot_valuefunc_and_policy(mc.Q)
    # Save the agent as pickle
    pickle.dump(mc, open('output/mc.pkl', 'wb'))

    # Run an episode after training
    # mc.generate_episode(render=True)