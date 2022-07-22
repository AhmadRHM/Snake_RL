import cv2

from env import SnakeEnv
from DQNAgent import DQNAgent
import numpy as np
import torch
import matplotlib.pyplot as plt

SHOW_PREVIEW = True
MODEL_NAME = "2conv"
AGGREGATE_STATS_EVERY = 20
epsilon = 0.01

if __name__ == '__main__':
    env = SnakeEnv(10)

    PATH = f'models/2conv___600.00max__160.68avg____0.00min__1658520478.model'
    agent = DQNAgent(env, PATH)

    ep_rewards = []
    visualization_data = {"episode": [], "loss": [], "avg": [], "min": [], "max": []}
    episode = 0
    while True:
        episode += 1

        episode_reward = 0
        current_state = env.reset()

        done = False
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_qs(current_state))
            else:
                action = np.random.randint(0, env.ACTION_N)
            new_state, reward, done = env.step(action)

            episode_reward += reward

            if SHOW_PREVIEW:
                env.render()

            current_state = new_state

        if SHOW_PREVIEW:
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
