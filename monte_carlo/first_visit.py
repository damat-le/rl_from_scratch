import gym
import numpy as np 

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
        #self.returns = {}

    # def initialise_policy(self):
    #     """
    #     Initialise the policy to a random policy.
    #     """
    #     policy={}

    #     for state in range(self.ns):
    #         for action in range(self.na):
    #             policy[state] = random.randint(0, self.na-1)
    #     return policy

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

    # def initialise_value_function(self):
    #     """
    #     Initialise the policy to a random policy.
    #     """
    #     Q={}

    #     for state in range(self.ns):
    #         for action in range(self.na):
    #             Q[(state, action)] = 0
    #     return Q

    def initialise_value_function(self):
        """
        Initialise the value function to zero.
        """
        Q = np.zeros((self.na, self.ns))
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
            episode.append((state, action, reward, next_state))
            
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

    
    def train(self, n_episodes: int = 1000):
        """
        Train the policy using the first-visit Monte Carlo method.
        """
        for episode in range(n_episodes):
            episode = self.generate_episode(verbose=False)
            #episode_len = len(episode)
            G = 0
            returns = self.initialise_returns()
            
            while episode:
                state, action, reward, next_state = episode.pop()
                G = self.gamma * G + reward
                
                #state = next_state
                if (state, action) not in {(s,a) for s,a,_,_ in episode}:
                    returns[(action, state)].append(G)
                    self.Q[action, state] = np.mean(returns[(action, state)])

                    a_greedy = np.argmax(self.Q[:,state])
                    self.policy[:,state] = self.eps / (self.na - 1)
                    self.policy[a_greedy, state] = 1 - self.eps


if __name__=='__main__':
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pickle
    import os

    # create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    # Train the agent
    env = gym.make('FrozenLake8x8-v1', is_slippery=False)
    mc = FirstVisitMC(env, gamma=.999, eps=.3)
    mc.train(n_episodes=1000000)
    v = mc.Q.max(axis=0).reshape(8,8)
    sns.heatmap(v, cmap='YlGnBu', annot=True, fmt='.2f')
    
    # Save the trained agent
    plt.savefig('output/v.png')
    pickle.dump(mc, open('output/mc.pkl', 'wb'))

    # Run an episode after training
    # mc.generate_episode(render=True)