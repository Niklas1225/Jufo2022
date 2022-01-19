from os import listdir, name
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

def loadDataAndAssemble(csv : boolean, name):
    if csv:
        insider_data = pd.read_csv(r"C:\Users\jensb\Documents\Praktikum\Stock_Prediction\03-07-2021\JugendForscht\CSVs\AMZN_insider.csv")
        insider_data = insider_data.drop(columns=["Unnamed: 0", "Instrument"])
        insider_data = insider_data[::-1]

        stock_data_part1 = pd.read_csv(r"C:\Users\jensb\Documents\Praktikum\Stock_Prediction\03-07-2021\JugendForscht\CSVs\AMZN1.csv", parse_dates=True)
        stock_data_part2 = pd.read_csv(r"C:\Users\jensb\Documents\Praktikum\Stock_Prediction\03-07-2021\JugendForscht\CSVs\AMZN2.csv", parse_dates=True)
        stock_data = pd.concat([stock_data_part1, stock_data_part2], ignore_index=True)
        stock_data = stock_data.reset_index().drop(columns="index")

    return insider_data, stock_data

insider_data, stock_data = loadDataAndAssemble(True, "AMZN")
insider_data = insider_data.dropna(thresh=3)
insider_data = insider_data.reset_index().drop(columns="index")

#get the target data
ema = stock_data["CLOSE"].ewm(span=20, adjust=False).mean()
stock_data["EMA"] = ema


print(stock_data.head(10))

X = insider_data
y = []
holder = 0
changed=False
for i in X["Insider Transaction Date"]:
    changed = False
    for a in range(holder, len(stock_data)):
        if stock_data["Date"][a] == i:
            y.append(stock_data["Date"][a])
            holder = a
            changed = True
            break
        if changed == False and a == len(stock_data)-1:
            print(i)
            exit()

#print(y[:-30])
#print(X.tail(30))

