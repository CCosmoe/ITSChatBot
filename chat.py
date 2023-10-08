import random
import json
import torch
from model import NeuralNetwork
from nltkfunctions import bag_of_words, tokenize
import tkinter as tk
from tkinter import messagebox

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents1.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = 'data.pth'
data = torch.load(FILE)                                                     #This is where everything is stored in from train_data.py. This is the trained model etc.

input_size = data['input_size']                                             #loads input size of model
hidden_size = data['hidden_size']                                           #loads hidden size of model
output_size = data['output_size']                                           #loads the output size of the model
all_words = data['all_words']                                               #loads all the words
tags = data['tags']                                                         #loads the tags
model_state = data['model_state']                                           #loads in the training model

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)      #model is going to do its calculation on GPU.
model.load_state_dict(model_state)
model.eval()


def get_response(user_input):

    sentence = tokenize(user_input)                         #sentence is tokenized just like its tokenized during training.
    X = bag_of_words(sentence, all_words)                   #passed into the bag_of_words function before passed into the model
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)                      #makes sure calculations are done on GPU

    output = model(X)                                       #returns the output of the model so it could be (1, 8) where 1 represents the input and predictions for each of 8 classes.
    tensormaxvalue, predicted = torch.max(output, dim=1)    #returns two values. We dont care about tensormaxvalue but we need the predicted as it indicates the tag.
    tag = tags[predicted.item()]                            #acquires the tag

    props = torch.softmax(output, dim=1)                    #softmax is applied to generate probability
    prob = props[0][predicted.item()]

    if prob.item() > 0.75:                                  #Needs to pass this threshold otherwise else statement is printed.
        for intent in intents['intents2']:
            if tag == intent['tag']:                        #if tags match print appropriate response.
                return random.choice(intent['responses'])
    else: 
        return "I am sorry I am having issues understanding. Please contact the ITS Service Desk at: 805-756-7000)"

def send_message():
    user_input = entry.get()
    if user_input == "quit":
        display_message(f"User: {user_input}")
        window.quit()
        return
    display_message(f"User: {user_input}")
    response = get_response(user_input)
    display_message(f"ITS Chatbot: {response}")
    entry.delete(0, tk.END)  # Clear the input field

def display_message(message):
    text_area.insert(tk.END, message + "\n")
    text_area.see(tk.END)  # Scroll to the end of the text area

# Create the main window
window = tk.Tk()

# Create a Text widget to display the chat messages
text_area = tk.Text(window, height=20, width=50)
text_area.pack()
# Create an Entry widget for user input
entry = tk.Entry(window, width=50)
entry.pack()

display_message(f"ITS Chatbot: Let's chat! type 'quit' to exit")
display_message(f"Here is what I can help you with")
display_message(f"- Duo")
display_message(f"- Tech Rentals")
display_message(f"- Eduroam Wifi")
display_message(f"- Setup VPN")
display_message(f"- Software")
# Bind the Enter key to the send_message function
window.bind('<Return>', lambda event: send_message())

# Create a button to send the user input
send_button = tk.Button(window, text="Send", command=send_message)
send_button.pack()

# Run the Tkinter event loop
window.mainloop()


# bot_name = "ITS BOT"
# print("Let's chat! type 'quit' to exit")
# print("Here is what I can help you with")
# print('- Duo')
# print('- Tech Rentals')
# print('- Eduroam Wifi')
# print('- Setup VPN')
# print('- Software')


# while True:


#     sentence = input('You: ')
#     if sentence == "quit": 
#         break       
#     sentence = tokenize(sentence)                           #sentence is tokenized just like its tokenized during training.
#     X = bag_of_words(sentence, all_words)                   #passed into the bag_of_words function before passed into the model
#     X = X.reshape(1, X.shape[0])
#     X = torch.from_numpy(X).to(device)                      #makes sure calculations are done on GPU

#     output = model(X)                                       #returns the output of the model so it could be (1, 8) where 1 represents the input and predictions for each of 8 classes.
#     tensormaxvalue, predicted = torch.max(output, dim=1)    #returns two values. We dont care about tensormaxvalue but we need the predicted as it indicates the tag.
#     tag = tags[predicted.item()]                            #acquires the tag

#     props = torch.softmax(output, dim=1)                    #softmax is applied to generate probability
#     prob = props[0][predicted.item()]

#     if prob.item() > 0.75:                                  #Needs to pass this threshold otherwise else statement is printed.
#         for intent in intents['intents2']:
#             if tag == intent['tag']:                        #if tags match print appropriate response.
#                 print(f"{bot_name}: {random.choice(intent['responses'])}")
#     else: 
#         print(f"{bot_name}: I am sorry I am having issues understanding. Please contact the ITS Service Desk at: 805-756-7000")
