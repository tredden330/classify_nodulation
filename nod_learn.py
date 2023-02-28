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
        self.input_layer    = nn.Linear(input_dim,9000)
        self.hidden_layer1  = nn.Linear(9000,250)
        self.hidden_layer2  = nn.Linear(250, 25)
        self.output_layer   = nn.Linear(25,output_dim)
        self.relu = nn.ReLU()

    def forward(self,x):
        out =  self.relu(self.input_layer(x))
        out =  self.relu(self.hidden_layer1(out))
        out =  self.relu(self.hidden_layer2(out))
        out =  self.output_layer(out)
        return out

#track program runtime
start_time = time.time()

#run on cpu/gpu(cuda)
device = 'cuda'

df1 = pd.read_csv('data/genes_with_float.csv', on_bad_lines='skip')

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

#split dataset into batches
batch_size = 50000
batches_X_train = torch.split(X_train, batch_size)
batches_y_train = torch.split(y_train, batch_size)
batches_X_test = torch.split(X_test, batch_size)
batches_y_test = torch.split(y_test, batch_size)



def train_network(model,optimizer,criterion,X_train,y_train,X_test,y_test,num_epochs):

    for epoch in range(num_epochs):

        epoch_percent_correct = []
        train_losses = []

        for batch_num in range(5):

            #clear out the gradients from the last step loss.backward()
            optimizer.zero_grad()

            #shuffle training data each epoch
            indices = torch.randperm(batches_X_train[batch_num].size()[0])
            shuffled_X_train = batches_X_train[batch_num][indices]
            shuffled_y_train = batches_y_train[batch_num][indices]

            #forward feed
            output_train = model(shuffled_X_train)

            #calculate the loss and record it for the batch
            loss_train = criterion(output_train, shuffled_y_train)
            train_losses.append(loss_train.item())

            #backward propagation: calculate gradients
            loss_train.backward()

            #update the weights
            optimizer.step()

            #calculate and validate predictions on test set for this batch
            test_predictions = model(batches_X_test[batch_num])
            count = 0
            correct = 0
            incorrect = 0
            for prediction in test_predictions:
                if np.argmax(prediction.detach().cpu().numpy()) == y_test[count].cpu().numpy():
                    correct += 1
                else:
                    incorrect += 1
                    count += 1
            epoch_percent_correct.append((correct/(correct+incorrect))*100)

        #print information for current epoch
        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("\t Percentage correct: ", epoch_percent_correct)
            print("\t Training Error: ", train_losses)

train_network(model,optimizer,criterion,X_train,y_train,X_test,y_test,num_epochs)

print("finished in ", time.time() - start_time, " seconds")