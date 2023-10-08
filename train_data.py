import json 
from nltkfunctions import tokenize, stemmy, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNetwork


with open('intents1.json', 'r') as f:
    intents = json.load(f)


all_words = []
tags = []
xy = []
for intent in intents['intents2']:      #Iterate over all intents index by index
    tag = intent['tag']                 #Gets the tag for each index
    tags.append(tag)                    #Appends the tag to tag list to append it later for to xy.
    for pattern in intent['patterns']:  #For each intent index iterate over all patterns
        splitting = tokenize(pattern)   #Splits each sentence such as How are you -> How, are, you
        all_words.extend(splitting)     #Adds the split sentence to all_words list.
        xy.append((splitting, tag))    #Appends both the split sentence and tag.

all_words = stemmy(all_words)       #Lowercase all the words and remove punctuations. Make it easier for computer to recognize patterns.
print(all_words)
all_words = sorted(set(all_words)) # removes duplicates from the list and returns a list again.
print(all_words)
tags = sorted(set(tags))            #remove the duplicate tags.

X_train = [] 
y_train = []
for(pattern_sentence,tag) in xy: 
    bag = bag_of_words(pattern_sentence, all_words) # The function returns a list with 0s and 1s. 0 represents the word not in allwords and 1 represents the word that is in all words.
    X_train.append(bag)                             # This appends the bag of words into the X_train list.and

    label = tags.index(tag)                         # This returns the index from the list of tags.
    y_train.append(label)                           # This appends the index to the y_train list.

class ChatDataset(Dataset):                                                                     #initiazling the Dataset class.
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]                                                                                                                                               
    
    def __len__(self):
        return self.n_samples

#hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 400
print(input_size, len(all_words))
print(output_size, tags)


dataset = ChatDataset()                                                                         #creating dataset
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)  #Uses train loader for efficiently accessing data.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                           #checks if cpu or gpu is avaliable
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)                          #creates a neural network model with its input size, hidden size, and output size and does calculation on cpu or gpu.

criterion = nn.CrossEntropyLoss()                                                               #This will be needed for calculating loss later.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)                              #initializes model parameters and its learning rate                                                                                                                                                                                                                                                         

for epoch in range(num_epochs):                                                                 #Number of epochs to eliminate bias.
    for(words, labels) in train_loader: 
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)                                         #gets the tag and passes it on to GPU
        outputs = model(words)                                                                  #passes the input into the model.
        loss =  criterion(outputs,labels)                                                       #calculating the loss
        optimizer.zero_grad()                                                                   #sets the gradient to zero to prevent accumulation.
        loss.backward()                                                                         #calculates the gradients so weights and bias can be updated.
        optimizer.step()                                                                        #updating the weights and bias. Back propagation
    if(epoch + 1) % 100 == 0: 
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')                          #printing loss for every 100 epoch

print(f'final loss,loss={loss.item():.4f}')                                                     #printing final epoch

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = 'data.pth'                                                                              #wrapping the whole thing in data.pth so it can be used with user's input later.
torch.save(data,FILE)
print(f'training complete. file saved to {FILE}')