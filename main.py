import torch
from torch import nn
from torch.optim import Adam
from tile3 import IHT, tiles
import random


class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 100)
        self.linear2 = nn.Linear(100, 50)
        self.linear3 = nn.Linear(50, 10)
        self.linear4 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, feature):
        output = self.linear1(feature)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.relu(output)
        output = self.linear3(output)
        output = self.relu(output)
        output = self.linear4(output)
        return output


class Agent:
    def __init__(self):
        self.learning_rate = 3e-4
        self.tiling_size = 4096
        self.num_tilings = 8

        self.terminal_position = torch.tensor(0.5)
        self.actions = torch.tensor([-1, 0, 1])
        self.epsilon = 0.1

        self.iht = IHT(self.tiling_size)

        self.model = NeuralNet(self.tiling_size * self.num_tilings)
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr= self.learning_rate)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    def initiate_episode(self):
        r2, r1 = -0.4, -0.6
        position = (r1 - r2) * torch.rand(size=(1,)) + r2
        velocity = torch.tensor(0)
        return position, velocity
    
    def get_feature(self, position, velocity, action):
        indices = tiles(self.iht,8,[8*position/(0.5+1.2),8*velocity/(0.07+0.07)],[action])
        feature = torch.zeros(self.tiling_size * self.num_tilings)
        for i in range(self.num_tilings):
            index = i * self.tiling_size + indices[i]
            feature[index] = 1
        return feature
    
    def get_action_value(self, position, velocity, action):
        if position != self.terminal_position:
            feature = self.get_feature(position, velocity, action).to(self.device)
            value = self.model(feature)
            return value
        else:
            return torch.tensor(0)
        
    def select_action(self, position, velocity):
        temp_val = torch.rand(size=(1,))
        if (temp_val < self.epsilon):
            return torch.randint(-1, 2, size=(1,))
        else:
            action_values = torch.tensor([0, 0, 0])
            for i in range(len(self.actions)):
                value = self.get_action_value(position, velocity, self.actions[i])
                action_values[i] = value
            optimal_actions = []
            for i in range(len(self.actions)):
                if action_values[i] == torch.max(action_values):
                    optimal_actions.append(i-1)
            return torch.tensor(random.choice(optimal_actions))

    def execute_action(self, curr_position, curr_velocity, action):
        next_velocity = curr_velocity.to(self.device)
        next_velocity = torch.add(next_velocity, torch.mul(torch.tensor(0.001), action))
        temp = torch.mul(torch.tensor(-0.0025), torch.cos(torch.mul(3, curr_position))).to(self.device)
        next_velocity = torch.add(next_velocity, temp)
        next_velocity = torch.clamp(next_velocity, min=-0.07, max=0.07)

        next_postion = torch.add(curr_position, next_velocity).to(self.device)
        next_postion = torch.clamp(next_postion, min=-1.2, max=0.5)

        if next_postion == torch.tensor(-1.2):
            next_velocity = torch.tensor(0)

        return torch.tensor(-1), next_postion, next_velocity


    def ql_update(self, curr_position, curr_velocity, action, reward, next_position, next_velocity):
        action_values = torch.tensor([0, 0, 0])
        for i in range(len(self.actions)):
            value = self.get_action_value(next_position, next_velocity, self.actions[i])
            action_values[i] = value
        max_action_value = torch.max(action_values)
        target = torch.add(reward, max_action_value)

        target = target.to(self.device)
        curr_action_value = self.get_action_value(curr_position, curr_velocity, action)
        batch_loss = self.criterion(curr_action_value, target.float())

        self.model.zero_grad()
        batch_loss.backward()
        self.optimizer.step()

    def ql_episode(self):
        time = 0
        curr_position, curr_velocity = self.initiate_episode()
        while curr_position != self.terminal_position:
            action = self.select_action(curr_position, curr_velocity)
            reward, next_position, next_velocity = self.execute_action(curr_position, curr_velocity, action)
            self.ql_update(curr_position, curr_velocity, action, reward, next_position, next_velocity)
            curr_position, curr_velocity = next_position, next_velocity
            time += 1
        return time


agent = Agent()
for i in range(1000):
    time = agent.ql_episode()
    file = open("process.txt", "a")
    file.write(str(i+1) + ", " + str(time))