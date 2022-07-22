import numpy as np
import random
import cv2
import time


class SnakeEnv:
    DIRECTIONS = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    COLORS = {"snake-head": (0, 75, 150), "snake-body": (55, 150, 250), "food": (0, 255, 0)}
    NN_COLORS = {"snake-head": -1, "snake-body": -0.5, "food": 1}

    COMPLETE_BONUS = 1000
    FOOD_BONUS = 30
    FAIL_PENALTY = 0
    MOVE_PENALTY = 0

    ACTION_N = 3

    def __init__(self, board_size):
        self.board_size = board_size
        self.snake_locations = None
        self.snake_direction = 0
        self.food_location = None

    def reset(self):
        self.snake_locations = [(self.board_size // 2, self.board_size // 2), (self.board_size // 2 + 1, self.board_size // 2)]
        self.snake_direction = 0
        self.food_location = self.spawn_new_food()
        return self.get_img()

    def get_img(self, get_real_image=False):
        if get_real_image:
            img = np.zeros((self.board_size, self.board_size, 3), dtype=np.uint8)
            img[self.snake_locations[0]] = self.COLORS["snake-head"]
            img[self.food_location] = self.COLORS["food"]
        else:
            img = np.zeros((self.board_size, self.board_size))
            img[self.snake_locations[0]] = self.NN_COLORS["snake-head"]
            img[self.food_location] = self.NN_COLORS["food"]
        for location in self.snake_locations[1:]:
            if get_real_image:
                img[location] = self.COLORS["snake-body"]
            else:
                img[location] = self.NN_COLORS["snake-body"]

        # to make x, y meaningful :)
        if get_real_image:
            img = img.transpose((1, 0, 2))
            img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
        else:
            img = img.T
        return img

    def render(self):
        img = self.get_img(True)
        cv2.imshow("snake!", img)
        cv2.waitKey(100)

    def get_free_squares(self):
        free_squares = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                if (x, y) not in self.snake_locations:
                    free_squares.append((x, y))
        return free_squares

    def spawn_new_food(self):
        free_squares = self.get_free_squares()
        return random.choice(free_squares)

    def turn_snake_right(self):
        self.snake_direction += 1
        self.snake_direction %= 4

    def turn_snake_left(self):
        self.snake_direction += -1 + 4
        self.snake_direction %= 4

    def step(self, action):  # action = 0: turn right 1: go straight 2: turn left ### returns new_state, reward, done
        if action == 0:
            self.turn_snake_right()
        elif action == 2:
            self.turn_snake_left()
        prev_x, prev_y = self.snake_locations[0]
        dx, dy = self.DIRECTIONS[self.snake_direction]
        new_x, new_y = prev_x + dx, prev_y + dy

        if 0 <= new_x < self.board_size and 0 <= new_y < self.board_size and (new_x, new_y) not in self.snake_locations[:-1]:
            if self.food_location == (new_x, new_y):
                self.snake_locations = [(new_x, new_y)] + self.snake_locations
                reward = self.FOOD_BONUS
                self.food_location = self.spawn_new_food()
                done = len(self.snake_locations) == (self.board_size**2)
                if done:
                    reward = self.COMPLETE_BONUS
            else:
                done = False
                self.snake_locations = [(new_x, new_y)] + self.snake_locations[:-1]
                reward = -self.MOVE_PENALTY
        else:
            done = True
            reward = -self.FAIL_PENALTY
        return self.get_img(), reward, done


if __name__ == '__main__':
    env = SnakeEnv(10, True)
    img = env.reset()
    actions = [0, 1, 1, 1, 1, 2]
    done = False
    while not done:
        cv2.imshow("snake!", cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST))
        k = cv2.waitKey()
        if k == ord('d'):
            action = 0
        elif k == ord('a'):
            action = 2
        else:
            action = 1
        # action = random.choice(actions)

        img, reward, done = env.step(action)
        print(env.DIRECTIONS[env.snake_direction])
        if action == 0:
            print("turning right")
        elif action == 1:
            print("going straight")
        else:
            print("turning left")
        print(reward)
