if __name__=='__main__':
    import pickle
    from first_visit import FirstVisitMC
    import gym
    import gym_simplegrid
    env = gym.make('SimpleGrid-8x8-v0')
    mc = pickle.load(open('output/mc.pkl', 'rb'))
    mc.env = env
    mc.generate_episode(render=True, verbose=True)