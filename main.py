import cv2

from env import SnakeEnv
from DQNAgent import DQNAgent
from tqdm import tqdm
import numpy as np
import time
import os
import torch
import matplotlib.pyplot as plt

epsilon = 0.9
MIN_EPSILON = 0.001
EPSILON_DECAY = 0.9995
EPISODES = 50_000
AGGREGATE_STATS_EVERY = 500
SHOW_PREVIEW = True
MIN_REWARD = 90
MODEL_NAME = "2conv"

if __name__ == '__main__':
    os.makedirs("models", exist_ok=True)

    env = SnakeEnv(10)
    agent = DQNAgent(env)

    ep_rewards = []
    visualization_data = {"episode": [], "loss": [], "avg": [], "min": [], "max": []}
    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        episode_reward = 0
        current_state = env.reset()

        done = False
        while not done:

            # This part stays mostly the same, the change is to query a model for Q values
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, env.ACTION_N)

            new_state, reward, done = env.step(action)

            episode_reward += reward

            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()

            # Every step we update replay memory and train main network
            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done)
            current_state = new_state

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            cv2.destroyAllWindows()

        ep_rewards.append(episode_reward)

        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

            visualization_data["episode"].append(episode)
            visualization_data["avg"].append(average_reward)
            visualization_data["min"].append(min_reward)
            visualization_data["max"].append(max_reward)

            plt.plot(visualization_data["episode"], visualization_data["avg"], label="average reward")
            plt.plot(visualization_data["episode"], visualization_data["min"], label="min reward")
            plt.plot(visualization_data["episode"], visualization_data["max"], label="max reward")
            plt.legend(loc=4)
            plt.show()

            print(epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if average_reward >= MIN_REWARD:
                PATH = f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model'
                torch.save(agent.model.state_dict(), PATH)

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
