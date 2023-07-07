import torch
import random

action_values = torch.tensor([0, 2, 2])
optimal_actions = []
for i in range(3):
    if action_values[i] == torch.max(action_values):
        optimal_actions.append(i-1)
print(torch.tensor(random.choice(optimal_actions)))