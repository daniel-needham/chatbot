import random
import json
import torch
from model import NeuralNet
from nltk_utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # will run on nvidia gpu if available

with open('intents.json', 'r') as f:
    intents = json.load(f)


FILE = "data.pth"
data = torch.load(FILE)

input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']
mode_state = data['model_state']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(mode_state)
model.eval() #sets the model to evaluation mode

bot_name = "Frodo"
print(f"Hi I'm {bot_name} Let's chat!, type 'quit' to exit")
while True:
    sentence = input('You: ')
    if sentence == 'quit':
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X) #converts from numpy array to tensor

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    probs = probs[0][predicted.item()]

    if probs.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f'{bot_name}: {random.choice(intent["responses"])}')

    else:
        print(f'{bot_name}: Haha! Weather in the Shire is a fickle thing.')
