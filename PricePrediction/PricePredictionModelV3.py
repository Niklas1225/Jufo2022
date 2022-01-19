from os import listdir
import random
from xmlrpc.client import boolean
from numpy.core.records import array
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.modules import loss
import eikon as ek
import math
from threading import Event
from datetime import datetime, date
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

def getDataFromCSV(path):
    data = pd.read_csv(path, index_col="Date", parse_dates=True)
    return data

df = getDataFromCSV(r"C:\Users\jensb\Documents\Praktikum\Stock_Prediction\03-07-2021\JugendForscht\CSVs\SPX.csv")

X = df.iloc[:-1, :]
y = pd.DataFrame(df["CLOSE"][1:])
mm = MinMaxScaler(feature_range=(0,1))
ss = StandardScaler()

X_ss = ss.fit_transform(X)
y_mm =mm.fit_transform(y)

X_batches = []
y_batches = []
for i in range(40, len(X_ss)):
    X_batches.append(X_ss[i-20:i])
    y_batches.append(y_mm[i-1, :])

X_batches = np.array(X_batches)
y_batches = np.array(y_batches)

X_train = X_batches[:1600, :]
X_test = X_batches[1600:, :]

y_train = y_batches[:1600, :]
y_test = y_batches[1600:, :]

#print("Training Shape: ", X_train.shape, y_train.shape)
#print("Testing Shape: ", X_test.shape, y_test.shape) 

X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test)) 

print("Training Tensor Shape: ", X_train_tensors.shape, y_train_tensors.shape)
print("Testing Tensor Shape: ", X_test_tensors.shape, y_test_tensors.shape) 

X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1,X_train_tensors.shape[1], X_train_tensors.shape[2]))
X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1,X_test_tensors.shape[1], X_test_tensors.shape[2]))

class Netz(nn.Module):
    def __init__(self, outputs, inputs, hiddens, layers, seq_length):
        super(Netz, self).__init__()
        self.outputs = outputs
        self.inputs = inputs
        self.hiddens = hiddens
        self.layers = layers
        self.seq_length = seq_length

        self.lstm1 = nn.LSTM(input_size=inputs, hidden_size=hiddens, num_layers=layers, batch_first=True)
        self.fc1 = nn.Linear(hiddens, 128)
        self.fc2 = nn.Linear(128, outputs)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        h_0 = self.initHiddenInternal(x)
        c_0 = self.initHiddenInternal(x)
        output, (hn, cn) = self.lstm1(x, (h_0, c_0))
        hn = hn[-1]
        hn = hn.view(-1, self.hiddens)
        out = self.relu(hn)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def initHiddenInternal(self, x):
        return Variable(torch.zeros(self.layers, x.size(0), self.hiddens))

num_epochs = 140
learning_rate = 0.0005

inputs = 4
hiddens = 30
layers = 2
outputs = 1

model = Netz(outputs, inputs, hiddens, layers, X_train_tensors.shape[0])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

lossData = []
def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model.forward(X_train_tensors)
    loss = criterion(output, y_train_tensors)
    loss.backward()
    optimizer.step()
    lossData.append(loss.item())
    print("Epoch: %d, loss: %1.5f" % (epoch + 1, loss.item()))

for i in range(num_epochs):
    train(i)

def test(plotting : boolean):
    #Test our Model with Test Data
    optimizer.zero_grad()
    train_predict = model(X_test_tensors)
    loss = criterion(train_predict, y_test_tensors)
    print("Loss: ", loss.item())
    train_predict = mm.inverse_transform(train_predict.data.numpy()).flatten().tolist()
    a = [np.nan] * (len(df) - len(train_predict))
    train_predict = a + train_predict
    df["Predicted Testing Data"] = train_predict
    if plotting:
        plt.figure(figsize=(15,10))
        plt.plot(df["CLOSE"], label="Actual Data")
        plt.plot(df["Predicted Testing Data"], label="Predicted Data")
        plt.legend()
        plt.show()



def plotLoss():
    plt.figure(figsize=(15, 10))
    plt.plot(lossData, label="Loss per epoch")
    plt.legend()
    plt.show()

test(plotting=True)
plotLoss()
