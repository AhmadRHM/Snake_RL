import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from collections import deque
from env import SnakeEnv
import random

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)


class Network(nn.Module):
    def __init__(self, board_size, action_n):
        super().__init__()
        linear_input_size = ((board_size - 2) // 2) - 2
        self.seq = nn.Sequential(nn.Conv2d(1, 16, 3),
                                 nn.ReLU(),
                                 nn.MaxPool2d(2, 2),
                                 nn.Conv2d(16, 25, 3),
                                 nn.ReLU(),
                                 nn.Flatten(),
                                 nn.BatchNorm1d(linear_input_size ** 2 * 25))
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(linear_input_size ** 2 * 25 + board_size**2 + 2, 64)
        self.fc2 = nn.Linear(64, action_n)

    def forward(self, x):
        boards = np.array(x)
        v3 = []
        for board in boards[:, 0, :, :]:
            head_x, head_y = np.where(board == -1)[0][0], np.where(board == -1)[1][0]
            food_x, food_y = np.where(board == 1)[0][0], np.where(board == 1)[1][0]
            v3.append(np.array([food_x - head_x, food_y - head_y]))
        v3 = torch.tensor(np.array(v3)).float()

        v1 = self.seq(x)
        v2 = torch.flatten(x, 1)

        vc = torch.cat((v1, v2, v3), 1)
        vc = self.dropout(vc)

        o_fc1 = F.relu(self.fc1(vc))
        return self.fc2(o_fc1)


class DQNAgent:
    def __init__(self, env: SnakeEnv):
        self.env = env

        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.update_target_model()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

    def create_model(self):
        return Network(self.env.board_size, self.env.ACTION_N).float()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Trains main network every step during episode
    def train(self, terminal_state):

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        self.optimizer.zero_grad()

        # Get current states from minibatch, then query NN model for Q values
        current_states = torch.tensor(np.array([transition[0] for transition in minibatch])).unsqueeze(1).float()
        current_qs_list = self.model(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = torch.tensor(np.array([transition[3] for transition in minibatch])).unsqueeze(1).float()
        with torch.no_grad():
            future_qs_list = self.target_model(new_current_states)
            future_qs_list = future_qs_list.numpy()

        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index].detach().numpy().copy()
            current_qs[action] = new_q
            # And append to our training data
            y.append(current_qs)
            # print(y[-1])

        y = torch.tensor(np.array(y)).float()

        # print(y)
        # print(current_qs_list)
        loss = F.mse_loss(current_qs_list, y)
        loss.backward()
        self.optimizer.step()

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.update_target_model()
            self.target_update_counter = 0

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        with torch.no_grad():
            self.model.eval()
            output = self.model(torch.tensor(np.array(state)).unsqueeze(0).unsqueeze(0).float())[0]
            self.model.train()
        return output.numpy()


if __name__ == '__main__':
    # net = Network(10, 3).float()
    #
    # inp = np.random.random((64, 1, 10, 10))
    # print(inp.shape)
    # tens = torch.tensor(inp)
    # print(tens.shape)
    # out = net(tens.float())
    # print(out.shape)

    env = SnakeEnv(10)
    agent = DQNAgent(env)
    print(agent.get_qs(np.random.random((10, 10))))

