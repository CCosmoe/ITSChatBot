import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)        #Defining 1st layer
        self.l2 = nn.Linear(hidden_size, hidden_size)       #Defining 2nd layer
        self.l3 = nn.Linear(hidden_size, num_classes)       #Defining 3rd layer
        self.relu = nn.ReLU()                               #activation function

    def forward(self, x): 
        out  = self.l1(x)               #1st layer
        out  = self.relu(out)           
        out  = self.l2(out)             #2nd layer
        out  = self.relu(out)
        out  = self.l3(out)             #3rd layer

        return out
