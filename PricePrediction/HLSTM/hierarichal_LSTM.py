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

#Shape and transform data



class Netz(nn.Module):
    def __init__(self, inputs_hourly, hiddens_hourly, outputs_hourly, layers_hourly, seq_length_hourly, bidirectional_hourly, inputs_daily, hiddens_daily, outputs_daily, layers_daily, seq_length_daily, bidirectional_daily):
        super(Netz, self).__init__()
        #Init variables for hourly / lower lstm
        self.outputs_hourly = outputs_hourly
        self.inputs_hourly = inputs_hourly
        self.hiddens_hourly = hiddens_hourly
        self.layers_hourly = layers_hourly
        self.seq_length_hourly = seq_length_hourly
        self.bidirectional_hourly = bidirectional_hourly
        #Init variables for daily / upper lstm
        self.outputs_daily = outputs_daily
        self.inputs_daily = inputs_daily
        self.hiddens_daily = hiddens_daily
        self.layers_daily = layers_daily
        self.seq_length_daily = seq_length_daily
        self.bidirectional_daily = bidirectional_daily
        
        #Init Layers
        #Lower Layer
        self.lstm_hourly = nn.LSTM(input_size=self.inputs_hourly, hidden_size=self.hiddens_hourly, num_layers=self.layers_hourly, batch_first=True, bidirectional= bidirectional_hourly)
        self.fc1 = nn.Linear(self.hiddens_hourly, 128)
        self.fc2 = nn.Linear(128, self.inputs_daily)

        #Upper Layer
        self.lstm_daily = nn.LSTM(input_size=self.inputs_daily, hiddens_size=self.hiddens_daily, num_layers=self.layers_daily, batch_first=False, bidirectional=bidirectional_daily)#batch_first = True? Check
        self.fc3 = nn.Linear(self.hiddens_daily, 128)
        self.fc4 = nn.Linear(128, self.outputs_daily)

        #Init activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    

    def forward(self, x_hourly, x_daily):
        h_0_hourly = self.initHiddenInternal(x_hourly, self.bidirectional_hourly)
        c_0_hourly = self.initHiddenInternal(x_hourly, self.bidirectional_hourly)

        output_hourly, (hn_hourly, cn_hourly) = self.lstm_hourly(x_hourly, (h_0_hourly, c_0_hourly))
        print(hn_hourly.shape)
        hn_hourly = hn_hourly[-1]
        weights_hourly = self.relu(hn_hourly)#Test other activation functions
        weights_hourly = self.fc1(weights_hourly)
        weights_hourly = self.fc2(weights_hourly)
        weights_hourly = self.relu(weights_hourly)#Test other activation functions

        #add weights_hourly to c_0_daily --> like shown in the diagramm from the link
        h_0_daily = self.initHiddenInternal(x_daily, self.bidirectional_daily)
        c_0_daily = self.initHiddenInternal(x_daily, self.bidirectional_daily)

        output_daily, (hn_daily, cn_daily) = self.lstm_daily(x_daily,(h_0_daily, c_0_daily))
        print(hn_daily.shape)
        hn_daily = hn_daily[-1]#Could i also use the last of the output_daily? Compare Loss
        out = self.relu(hn_daily)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

    def initHiddenInternal(self, x, bidirectional):
        d=1
        if bidirectional == True:
            d=2
        return Variable(torch.zeros(self.layers * d, x.size(0), self.hiddens))

num_epochs = 140
learning_rate = 0.0005

inputs_hourly = 4
hiddens_hourly = 30
layers_hourly = 2
outputs_hourly = 1
seq_length_hourly = X_hourly_tensors.shape[0]#Placeholder --> Put the real Tensor in
bidirectional_hourly = False

inputs_daily = 4
hiddens_daily = 30
layers_daily = 2
outputs_daily = 1
seq_length_daily = X_daily_tensors.shape[0]#Placeholder --> Put the real Tensor in
bidirectional_daily=False

model = Netz(inputs_hourly, hiddens_hourly, outputs_hourly, layers_hourly, seq_length_hourly, bidirectional_hourly, inputs_daily, hiddens_daily, outputs_daily, layers_daily, seq_length_daily, bidirectional_daily)

'''
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

lossData = []
def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model.forward(X_train_tensors)
    print(output.shape)
    print(y_train_tensors.shape)
    loss = criterion(output, y_train_tensors)
    loss.backward()
    optimizer.step()
    lossData.append(loss.item())
    print("Epoch: %d, loss: %1.5f" % (epoch + 1, loss.item()))

for i in range(num_epochs):
    train(i)
'''