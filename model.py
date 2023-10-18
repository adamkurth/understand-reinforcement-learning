import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

"""
    Coded by Adam Kurth
    Reference Tutorial: https://youtu.be/L8ypSXwyBds?si=h-6_idN4Wwsg_kSf
"""
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() #inherit from nn.Module
        self.linear1 = nn.Linear(input_size, hidden_size) 
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x): #( prediction )
        # apply linear layer and activation function 
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) #use adam optimizer
        self.criterion = nn.MSELoss() # use mean squared error loss function
        # (Q_new - Q)^2
        
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long) #long is integer
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x) 
        # n is number of samples in the batches 
        # x is the number of features in the sample
        
        if len(state.shape) == 1:
            # (1,x) # appends 1 dimension in the begnning
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, ) #tuple
        
        # 1. predicted Q values with current state
        pred = self.model(state)
        
        target = pred.clone() 
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
                # Q_new = R + gamma * max(next_predicted Q value)
            target[idx][torch.argmax(action[idx]).item()] = Q_new # Q_new is the target value
            
        # 2. r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone() is the predicted Q value
        # preds[argsmax(action)] = Q_new
        
        self.optimizer.zero_grad() # set all gradients to zero
        loss = self.criterion(target, pred) # loss function
        loss.backward() #backpropagation
        
        self.optimizer.step() #update the weights
        
        
        
        