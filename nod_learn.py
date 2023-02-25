import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import ast

class NeuralNetworkClassificationModel(nn.Module):

    def __init__(self,input_dim,output_dim):
        super(NeuralNetworkClassificationModel,self).__init__()
        self.input_layer    = nn.Linear(input_dim,900)
        self.hidden_layer1  = nn.Linear(900,25)
        self.output_layer   = nn.Linear(25,output_dim)
        self.relu = nn.ReLU()

    def forward(self,x):
        out =  self.relu(self.input_layer(x))
        out =  self.relu(self.hidden_layer1(out))
        out =  self.output_layer(out)
        return out

#track program runtime
start_time = time.time()

#run on cpu/gpu(cuda)
device = 'cpu'

df1 = pd.read_csv('data/genes_with_float.csv')

#extract x and y
X = df1['float_seq']
y = df1['nod_relation'].astype(int).values

#convert pandas series to float array 
X = X.tolist()
float_X = []
for seq in X:
    res = ast.literal_eval(seq)
    float_X.append(res)

X_train, X_test, y_train, y_test = train_test_split(float_X, y, test_size=0.2)

#turn into pytorch tensors
X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.LongTensor(y_train).to(device)
y_test = torch.LongTensor(y_test).to(device)

#initialize network
input_dim  = 300
output_dim = 2
model = NeuralNetworkClassificationModel(input_dim,output_dim).to(device)

#create optimizer and loss function object
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
num_epochs = 10

def train_network(model,optimizer,criterion,X_train,y_train,X_test,y_test,num_epochs):
    for epoch in range(num_epochs):
        #clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()

        #forward feed
        output_train = model(X_train)

        #calculate the loss
        loss_train = criterion(output_train, y_train)

        #backward propagation: calculate gradients
        loss_train.backward()

        #update the weights
        optimizer.step()

        #calculate and validate predictions on test set
        test_predictions = model(X_test)
        count = 0
        correct = 0
        incorrect = 0
        for prediction in test_predictions:
            if np.argmax(prediction.detach().cpu().numpy()) == y_test[count].cpu().numpy():
                correct += 1
            else:
                incorrect += 1
            count += 1

        #print information for current epoch
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"\t Train Loss: {loss_train.item():.4f}")
            print("\t Correct testing guesses: ", correct)
            print("\t Incorrect testing guesses: ", incorrect)
            print("\t Percentage correct: ", (correct/(correct+incorrect))*100)

train_network(model,optimizer,criterion,X_train,y_train,X_test,y_test,num_epochs)

print("finished in ", time.time() - start_time, " seconds")
