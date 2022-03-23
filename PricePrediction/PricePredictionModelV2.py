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
for i in range(60, len(X_ss)):
    X_batches.append(X_ss[i-60:i])
    y_batches.append(y_mm[i, :])

X_batches = np.array(X_batches)
y_batches = np.array(y_batches)

X_train = X_batches[:1500, :]
X_test = X_batches[1500:, :]

y_train = y_batches[:1500, :]
y_test = y_batches[1500:, :]

print("Training Shape: ", X_train.shape, y_train.shape)
print("Testing Shape: ", X_test.shape, y_test.shape) 

X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test)) 

X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1], X_train_tensors.shape[2]))
X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0],1, X_test_tensors.shape[1], X_test_tensors.shape[2]))

print("Training Tensor Shape: ", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Tensor Shape: ", X_test_tensors_final.shape, y_test_tensors.shape) 

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
        #print("x: ", x.shape)
        h_0 = self.initHiddenInternal(x)
        c_0 = self.initHiddenInternal(x)
        output, (hn, cn) = self.lstm1(x, (h_0, c_0))
        output = output.view(-1, self.hiddens)   
        out = self.relu(output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def initHiddenInternal(self, x):
        return Variable(torch.zeros(self.layers, x.size(0), self.hiddens))

num_epochs = 10
learning_rate = 0.0005

inputs = 4
hiddens = 6
layers = 2
outputs = 1

model = Netz(outputs, inputs, hiddens, layers, X_train_tensors_final.shape[1])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

lossData = []
def train(epoch):
    model.train()
    for i in range(X_train_tensors_final.shape[0]):
        optimizer.zero_grad()
        output = model.forward(X_train_tensors_final[i])
        loss = criterion(output, y_train_tensors[i])
        loss.backward()
        optimizer.step()
    lossData.append(loss.item())
    if epoch %1 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch + 1, loss.item()))

for i in range(num_epochs):
    train(i)

#Noch nicht überarbeitet
'''
def test(plotting : bool):
    loss_sum = 0
    #model.eval()
    for i in range(X_test_tensors_final.shape[0]):
        optimizer.zero_grad()
        predict = model(X_test_tensors_final[i])
        loss = criterion(predict, y_test_tensors[i])
        loss_sum += loss.item() / X_test_tensors_final.shape[0]
    print(loss_sum)

    if plotting:
        predict = mm.inverse_transform(predict.data.numpy()).flatten().tolist()
        a = [np.nan] * (len(df) - len(predict))
        predict = a + predict
        df["Predicted Testing Data"] = predict

        plt.figure(figsize=(15,10))
        plt.plot(df["CLOSE"], label="Actual Data")
        plt.plot(df["Predicted Testing Data"], label="Predicted Data")
        plt.legend()
        plt.show()

test(plotting=True)


def test():
    #Test our Model with Test Data
    optimizer.zero_grad()
    train_predict = model(X_test_tensors)
    loss = criterion(train_predict, y_test_tensors)
    print("Loss: ", loss.item())


    
    

    

    def plotPredictedData():
        plt.figure(figsize=(15,10))
        plt.plot(df["CLOSE"], label="Actual Data")
        plt.plot(df["Predicted Training Data"], label="Predicted Data")
        plt.legend()
        plt.show()

#plotPredictedData()
def plotLoss():
    plt.figure(figsize=(15, 10))
    plt.plot(lossData, label="Loss per 100 periods")
    plt.legend()
    plt.show()

plotLoss()
'''