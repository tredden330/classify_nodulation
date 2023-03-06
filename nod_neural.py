import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import ast
import statistics

class NeuralNetworkClassificationModel(nn.Module):

    def __init__(self,input_dim,output_dim):

        super(NeuralNetworkClassificationModel,self).__init__()

        #define shape of network
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 20),
            nn.ReLU(),
            nn.Linear(20,10),
            nn.ReLU(),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5,output_dim),
         )

    def forward(self,x):

        logits = self.linear_relu_stack(x)
        return logits



#track program runtime
start_time = time.time()

#run on cpu/gpu(cuda)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#choose between encoding schemes
isOneHotEncoded = True
isFloatEncoded = False

if (isOneHotEncoded):
    df1 = pd.read_csv('data/genes_with_onehot.csv', on_bad_lines='skip', nrows=100000)
#    df1 = df1.sample(frac=.01)
    keys = map(str, range(1200))
    X = df1[keys].values
    y = df1['nod_relation'].astype(int).values

if (isFloatEncoded):
    df1 = pd.read_csv('data/genes_with_float_million.csv', on_bad_lines='skip', nrows=200000)
    #extract x and y
    X = df1['float_seq']
    y = df1['nod_relation'].astype(int).values

    #convert pandas series to float array
    X = X.tolist()
    float_X = []
    for seq in X:
        res = ast.literal_eval(seq)
        float_X.append(res)
    X = float_X
    
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#turn into pytorch tensors
X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_train = torch.LongTensor(y_train).to(device)
y_test = torch.LongTensor(y_test).to(device)

#initialize network
input_dim  = 1200
output_dim = 2
model = NeuralNetworkClassificationModel(input_dim,output_dim).to(device)

#create optimizer and loss function object
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
num_epochs = 500

#split dataset into batches
batch_size = 100000
batches_X_train = torch.split(X_train, batch_size)
batches_y_train = torch.split(y_train, batch_size)
batches_X_test = torch.split(X_test, batch_size)
batches_y_test = torch.split(y_test, batch_size)



def train_network(model,optimizer,criterion,X_train,y_train,X_test,y_test,num_epochs):

    all_train_losses = []
    all_test_losses = []
    all_percent_correct = []

    for epoch in range(num_epochs):

        epoch_percent_correct = []
        epoch_train_losses = []
        epoch_test_losses = []

        #train model in batches
        for batch_num in range(len(batches_X_train)):

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
            epoch_train_losses.append(loss_train.item())

            #backward propagation: calculate gradients
            loss_train.backward()

            #update the weights
            optimizer.step()

        #test model in batches
        for batch_num in range(len(batches_X_test)):

            #calculate and validate predictions on test set for this batch
            test_predictions = model(batches_X_test[batch_num])

            #calculate the loss and record it for the batch
            loss_test = criterion(test_predictions, batches_y_test[batch_num])
            epoch_test_losses.append(loss_test.item())

            count = 0
            correct = 0
            incorrect = 0
            for prediction in test_predictions:
                if np.argmax(prediction.detach().cpu().numpy()) == batches_y_test[batch_num][count].cpu().numpy():
                    correct += 1
                else:
                    incorrect += 1
                count += 1

            percent_correct = (correct/(correct+incorrect))*100
            epoch_percent_correct.append(percent_correct)

        #print information for current epoch
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("\t Training Error: ", statistics.mean(epoch_train_losses))
            print("\t Testing Error: ", statistics.mean(epoch_test_losses))
            print("\t Percentage correct: ", statistics.mean(epoch_percent_correct))

        all_train_losses.append(statistics.mean(epoch_train_losses))
        all_test_losses.append(statistics.mean(epoch_test_losses))
        all_percent_correct.append(statistics.mean(epoch_percent_correct))
    return all_train_losses, all_test_losses, all_percent_correct

def graph(results):

    x = range(num_epochs)

    #graph errors over epochs
    fig, ax = plt.subplots()
    ax.plot(x, results[0], label="training error")
    ax.plot(x, results[1], label="testing error")
    plt.title("Neural Net - Average Error")
    plt.xlabel("training iteration")
    ax.grid()
    plt.legend()
    fig.savefig("nn_errors.png")

    #graph correct testing guesses over epochs
    fig, ax = plt.subplots()
    ax.plot(x, results[2], label="correct guess fraction")
    plt.title("Neural Net - Computer Guesses on Unseen Data")
    plt.xlabel("training iteration")
    plt.ylabel("percent")
    ax.grid()
    plt.legend()
    fig.savefig("nn_percent_correct.png")



results = train_network(model,optimizer,criterion,X_train,y_train,X_test,y_test,num_epochs)

graph(results)

torch.save(model.state_dict(), "nn_model.torch")

print("finished in ", time.time() - start_time, " seconds")
