import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.modules import loss
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
        self.lstm_hourly = nn.LSTM(input_size=self.inputs_hourly, hidden_size=self.hiddens_hourly, num_layers=self.layers_hourly, batch_first=True)
        self.fc1 = nn.Linear(self.hiddens_hourly, 128)
        self.fc2 = nn.Linear(128, self.hiddens_daily)

        #Output for lower lstm
        self.fc3 = nn.Linear(self.hiddens_daily, 1)#Testen ob es besser wäre diesen Parameter immer zu freezen, da dann nur die Layer für die Weights optimiert werden

        #Upper Layer
        self.lstm_daily = nn.LSTM(input_size=self.inputs_daily, hidden_size=self.hiddens_daily, num_layers=self.layers_daily, batch_first=True, bidirectional=bidirectional_daily)#batch_first = True? Check
        self.fc4 = nn.Linear(self.hiddens_daily, 128)
        self.fc5 = nn.Linear(128, self.outputs_daily)

        #Init activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

        #Try using module dicts
        self.lower_params = []
        self.lower_params.extend(self.lstm_hourly.parameters())
        self.lower_params.extend(self.fc1.parameters())
        self.lower_params.extend(self.fc2.parameters())
        self.lower_params.extend(self.fc3.parameters())

        self.upper_params = []
        self.upper_params.extend(self.lstm_daily.parameters())
        self.upper_params.extend(self.fc4.parameters())
        self.upper_params.extend(self.fc5.parameters())



    def forward(self, x_hourly, x_daily):
        h_0_hourly = self.initHiddenInternal(x_hourly, self.layers_hourly, self.hiddens_hourly, self.bidirectional_hourly)
        c_0_hourly = self.initHiddenInternal(x_hourly, self.layers_hourly, self.hiddens_hourly, self.bidirectional_hourly)

        output_hourly, (hn_hourly, cn_hourly) = self.lstm_hourly(x_hourly, (h_0_hourly, c_0_hourly))
        hn_hourly = hn_hourly[-1]
        weights_hourly = self.relu(hn_hourly)#Test other activation functions
        weights_hourly = self.fc1(weights_hourly)
        weights_hourly = self.fc2(weights_hourly)
        weights_hourly = self.relu(weights_hourly)#Test other activation functions

        output_hourly = self.fc3(weights_hourly)
        
        h_0_weights = []
        for i in range(weights_hourly.shape[0]):
            if i%16==0:
                h_0_weights.append(weights_hourly[i][:].data.tolist())

        h_0_weights = np.array(h_0_weights, np.float32)
        h_0_daily = Variable(torch.tensor(h_0_weights))
        h_0_daily_final = torch.reshape(h_0_daily, (1, h_0_daily.shape[0], h_0_daily.shape[1]))
        #print(h_0_daily_final.shape)

        c_0_daily = self.initHiddenInternal(x_daily, self.layers_daily, self.hiddens_daily, self.bidirectional_daily)

        output_daily, (hn_daily, cn_daily) = self.lstm_daily(x_daily,(h_0_daily_final, c_0_daily))

        hn_daily = hn_daily[-1]#Could i also use the last of the output_daily? Compare Loss
        out = self.relu(hn_daily)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        #print(out.shape)
        return output_hourly, out

    def initHiddenInternal(self, x, layers, hiddens, bidirectional):
        d=1
        if bidirectional == True:
            d=2
        return Variable(torch.zeros(layers * d, x.size(0), hiddens))

