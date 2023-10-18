
"""
    Coded by Adam Kurth
    Reference Tutorial: https://youtu.be/L8ypSXwyBds?si=h-6_idN4Wwsg_kSf
    
Context: This file contains the implementation of the snake game.
This file contains the implementation of a reinforcement deep neural network for the snake game. 
The network uses deep Q learning, where the q value represents the quality of an action. 
The network is trained to predict the best action to take based on the current state of the game. 
The training process involves updating the parameters of the network using the Bellman equation. 
The game environment is built in step 1 of the implementation.

# 11 different combination of boolean inputs, indicating danger (straight, left, right)
# Action is the output layer, where we return what the snake will do
# Training: (deep Q ;learning) - q value = quality of action.
# predictions become better over time, instead of becoming random variables.
#  according to bellman equation, the parameters are updated.

# simplified: Q = model.predict(state)
#              Q_new = R + gamma * max(Q(state_1))
# loss function = MSE = (Q_new - Q)^2

# step 1. Build the game (enviornment) (use pygame)
"""

import numpy as np
import random 
import torch    
from game import SnakeGameAI, Direction, Point
from collections import deque #datastructure to store memories
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000 # max number of experiences we want to store
BATCH_SIZE = 1000    # mini batch size for sampling
LR = 0.001           # learning rate

class Agent:
    
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # control randomness
        self.gamma = 0.9  # discount rate (<1)
        self.memory = deque(maxlen=MAX_MEMORY) #remove left elements (popleft)
        self.model = Linear_QNet(11, 256, 3) # 11 inputs, 256 hidden layer, 3 outputs
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # discount rate, trainer object to train the model
                
    def get_state(self, game):
        head = game.snake[0]                       #grab head from game.snake
        point_l = Point(head.x - 20, head.y) #create points in all directions to check for boundary
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        # directions
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN
        
        state = [
            #Dagner straight
            # checking in every direction whether the point that we made will hit the boundary of the game. 
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            
            #Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            
            #Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            
            #Move direction 
            # direction of the snake is left right etc == 1
            dir_l, 
            dir_r, 
            dir_u, 
            dir_d,
            
            # food location 
            game.food.x < game.head.x, #food is left of snake
            game.food.x > game.head.x, #food is right of snake
            game.food.y < game.head.y, #food is above snake
            game.food.y > game.head.y #food is below snake
            ]
        # converts boolean values to 0 or 1 
        return np.array(state, dtype=int)
                
    def remember(self, state, action, reward, next_state, done):
        # stores in memory
        self.memory.append((state, action, reward, next_state, done)) # exceeds max memory, removes left elements
        #stores as only one tuple
        
    def train_long_memory(self):
        # grab batch
        if len(self.memory) > BATCH_SIZE:
            # random sample from memory
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, done = zip(*mini_sample) # unzips into tuples
        self.trainer.train_step(states, actions, rewards, next_states, done)
        
        # Could alternatively use a for loop 
    
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)
            
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    
    def get_action(self, state):
        # random moves: tradeoff between exploration and exploitation
        # exploration: random moves
        # exploitation: best moves
        # better the model gets the more we exploit the best moves
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon: 
            # smaller epsilon gets the more we exploit the best moves
            move = random.randint(0,2) #(random move 0, 1, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() #move is integer
            final_move[move] = 1
            
        return final_move
            
def train():
    # for plotting
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0  
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state (current)
        state_old = agent.get_state(game)
        
        # get move
        final_move = agent.get_action(state_old)
        
        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        
        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # remember
        agent.remember(state_old, final_move, reward, state_new, done)
        # train long memory
        if done:
            # train long_memory (trains again on all previous moves for improvement overall)
            # and plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save()
                print('Game', agent.n_games, 'Score', score, 'Record', record)
        
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)   
            plot(plot_scores, plot_mean_scores)
            
        
    
if __name__ == '__main__':
    train()
    
    


