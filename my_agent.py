import numpy as np
import pygame
from pytorch_mlp import MLPRegression
import argparse
from console import FlappyBirdEnv
from collections import deque
import random

STUDENT_ID = 'a1785401'
DEGREE = 'UG'


class MyAgent:
    def __init__(self, show_screen=False, load_model_path=None, mode=None):
        # do not modify these
        self.show_screen = show_screen
        if mode is None:
            self.mode = 'train'  # mode is either 'train' or 'eval', we will set the mode of your agent to eval mode
        else:
            self.mode = mode

        # modify these
        self.storage = deque(maxlen=10000)
        self.network = MLPRegression(input_dim=6, output_dim=3, learning_rate=0.0005)
        # network2 has identical structure to network1, network2 is the Q_f
        self.network2 = MLPRegression(input_dim=6, output_dim=3, learning_rate=0.0005)
        # initialise Q_f's parameter by Q's, here is an example
        MyAgent.update_network_model(net_to_update=self.network2, net_as_source=self.network)

        self.epsilon = 1.0  # probability ε in Algorithm 2
        self.n = 64  # the number of samples you'd want to draw from the storage each time
        self.discount_factor = 0.99  # γ in Algorithm 2
        self.prev_score = 0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.tau = 0.01

        # do not modify this
        if load_model_path:
            self.load_model(load_model_path)

    def extract_features(self, state: dict) -> np.ndarray:

        bird_y = state['bird_y']
        bird_velocity = state['bird_velocity']
        screen_height = state['screen_height']
        pipes = state['pipes']

        if pipes:
            pipe = pipes[0]
            pipe_x = pipe.get('x', 400)
            pipe_mid = (pipe.get('top', 0) + pipe.get('bottom', 600)) / 2
            dx = pipe_x - state['bird_x']
            dy = pipe_mid - bird_y
        else:
            dx = 100
            dy = 0

        features = np.array([
            bird_y / screen_height,
            bird_velocity / 10,
            dx / 300,
            dy / 300,
            (screen_height - bird_y) / screen_height,
            bird_y / screen_height
        ], dtype = np.float32)

        return features



    def choose_action(self, state: dict, action_table: dict) -> int:
        """
        This function should be called when the agent action is requested.
        Args:
            state: input state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            action: the action code as specified by the action_table
        """
        # following pseudocode to implement this function

        features = self.extract_features(state).reshape(1,-1)
        if self.mode == 'train' and np.random.rand() < self.epsilon:
            a_t = np.random.choice(list(action_table.values())[:2])
        else:
            q_values = self.network.predict(features)
            a_t = int(np.argmax(q_values))

        self.prev_state = self.extract_features(state)
        self.prev_action = a_t

        return a_t

    def receive_after_action_observation(self, state: dict, action_table: dict) -> None:
        """
        This function should be called to notify the agent of the post-action observation.
        Args:
            state: post-action state representation (the state dictionary from the game environment)
            action_table: the action code dictionary
        Returns:
            None
        """
        # following pseudocode to implement this function
        
        new_state_vec = self.extract_features(state)
        done = state['done']
        reward = -100 if done else -1
        if state['score'] > self.prev_score:
            reward = +100
        self.prev_score = state['score']

        pipe_info = state.get('pipe_attributes', {})
        gap = pipe_info.get('gap', 0)
        formation = pipe_info.get('formation', '')

        if gap >= 500:
            level = 1
        elif gap == 150 and formation == 'random':
            level = 2
        elif formation == 'sine':
            level = 3
        else:
            level = 4  # for levels 4–6


        # for all levels

        

        if not done:
            dy_norm = new_state_vec[3]
            vel_norm = new_state_vec[1]
            prev_dy_norm = self.prev_state[3] if hasattr(self, 'prev_state') else dy_norm



        #level 1: no pipe
        
            if level == 1:
                reward += 5 * (1-abs(dy_norm))

            elif level == 2:
                reward += 8 * (1-abs(dy_norm))
                reward += 3 * (1-abs(vel_norm))
                reward += 4

            elif level == 3:
                reward += 8 * (1-abs(dy_norm))
                reward += 3 * (1-abs(vel_norm))
                reward += 4

                if abs(dy_norm) < abs(prev_dy_norm):
                    reward += 2
                elif abs(dy_norm) < abs(prev_dy_norm):
                    reward -= 2

            elif level == 4:
                reward += 10 * (1-abs(dy_norm))
                reward += 5 * (1-abs(vel_norm))

                if abs(dy_norm) < abs(prev_dy_norm):
                    reward += 3
                elif abs(dy_norm) > abs(prev_dy_norm):
                    reward -= 3

            elif level == 5:
                reward += 10 * (1-abs(dy_norm))
                reward += 5 * (1-abs(vel_norm))
            
                if abs(dy_norm) < abs(prev_dy_norm):
                    reward += 3
                elif abs(dy_norm) > abs(prev_dy_norm):
                    reward -= 3

                bird_y_norm = new_state_vec[0]
                edge_distance = min(bird_y_norm, 1-bird_y_norm)

                if edge_distance < 0.1:
                    reward -= 5

            elif level == 6:
                reward += 12 * (1-abs(dy_norm))
                reward += 8 * (1-abs(vel_norm))
            
                if abs (dy_norm) < abs (prev_dy_norm):
                    reward += 4
                elif abs(dy_norm) > abs(prev_dy_norm):
                    reward -= 4

                bird_y_norm = new_state_vec[0]
                edge_distance = min(bird_y_norm, 1 - bird_y_norm)
                if edge_distance < 0.1:
                    reward -= 5

        self.storage.append((self.prev_state, self.prev_action,reward, new_state_vec, done))

        if self.mode == 'train' and len(self.storage) > self.n:
            batch = random.sample(self.storage, self.n)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            next_states = np.array(next_states)
            dones = np.array(dones)

            next_q_values = self.network2.predict(next_states)
            max_next_q = np.max(next_q_values, axis=1)
            targets = rewards + self.discount_factor * max_next_q * (1 - dones.astype(int))

            q_values = self.network.predict(states)

            for i in range(self.n):
                q_values[i][actions[i]] = targets[i]
            W = np.ones_like(q_values)
            self.network.fit_step(states, q_values, W)

        for param, target_param in zip(self.network.parameters(), self.network2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        



    def save_model(self, path: str = 'my_model.ckpt'):
        """
        Save the MLP model. Unless you decide to implement the MLP model yourself, do not modify this function.

        Args:
            path: the full path to save the model weights, ending with the file name and extension

        Returns:

        """
        self.network.save_model(path=path)

    def load_model(self, path: str = 'my_model.ckpt'):
        """
        Load the MLP model weights.  Unless you decide to implement the MLP model yourself, do not modify this function.
        Args:
            path: the full path to load the model weights, ending with the file name and extension

        Returns:

        """
        self.network.load_model(path=path)

    @staticmethod
    def update_network_model(net_to_update: MLPRegression, net_as_source: MLPRegression):
        """
        Update one MLP model's model parameter by the parameter of another MLP model.
        Args:
            net_to_update: the MLP to be updated
            net_as_source: the MLP to supply the model parameters

        Returns:
            None
        """
        net_to_update.load_state_dict(net_as_source.state_dict())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, default=1)
    args = parser.parse_args()

    # bare-bone code to train your agent (you may extend this part as well, we won't run your agent training code)
    env = FlappyBirdEnv(config_file_path='config.yml', show_screen=True, level=args.level, game_length=50)
    agent = MyAgent(show_screen=True)
    episodes = 10000

    for episode in range(episodes):
        env.play(player=agent)
        

        # env.score has the score value from the last play
        # env.mileage has the mileage value from the last play
        print(env.score)
        print(env.mileage)

        # store the best model based on your judgement
        agent.save_model(path='my_model.ckpt')

        # you'd want to clear the memory after one or a few episodes
        ...

        # you'd want to update the fixed Q-target network (Q_f) with Q's model parameter after one or a few episodes
        ...
        if episode % 10 == 0:
            MyAgent.update_network_model(agent.network2, agent.network)
        
        agent.epsilon = max(0.1, agent.epsilon * 0.99)

    # the below resembles how we evaluate your agent
    env2 = FlappyBirdEnv(config_file_path='config.yml', show_screen=False, level=args.level)
    agent2 = MyAgent(show_screen=False, load_model_path='my_model.ckpt', mode='eval')

    scores = []
    for _ in range(10):
        env2.play(player=agent2)
        scores.append(env2.score)

    print(np.max(scores))
    print(np.mean(scores))
