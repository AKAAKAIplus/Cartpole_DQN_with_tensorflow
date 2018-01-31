# encoding: utf-8
import time
import gym
from double_network_DQN import double_network_DQN



env = gym.make('CartPole-v0') 
env = env.unwrapped

observation = env.reset()

DQN = double_network_DQN(num_of_actions=env.action_space.n, num_of_features=env.observation_space.shape[0])

count = 0
for num_of_episode in range(3000):

    observation = env.reset()
    ep_r = 0
    start = time.time()
    while True:
        env.render()
        
        
        action = DQN.action(observation)
        # calculate the reward
        observation_new, reward, done, info = env.step(action)

        x, x_dot, theta, theta_dot = observation_new



        x, x_dot, theta, theta_dot = observation_new
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2*1.2

        DQN.learning_buffers(observation, action, reward, observation_new)

        if  count > 1000:
        	DQN.learn()

        if done:
            end = time.time()
            print("Time taken: "+str(end - start)+"s")
            start = time.time()
            break

        observation = observation_new

        count += 1

        