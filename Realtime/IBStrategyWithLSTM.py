#Imports
import ibapi
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *
from numpy.core.arrayprint import set_printoptions
import ta
import numpy as np
import pandas as pd
import pytz
import math
from datetime import datetime, timedelta
import threading
import time
import torch
import torch.optim as optim

from HLSTM import Netz

#Vars
orderId = 1
#Class for Interactive Brokers Connection
class IBApi(EWrapper,EClient):
    def __init__(self):
        EClient.__init__(self, self)
    # Historical Backtest Data
    def historicalData(self, reqId, bar):
        try:
            bot.on_bar_update(reqId,bar,False)
        except Exception as e:
            print(e)
    # On Realtime Bar after historical data finishes
    def historicalDataUpdate(self, reqId, bar):
        try:
            bot.on_bar_update(reqId,bar,True)
        except Exception as e:
            print(e)
    # On Historical Data End
    def historicalDataEnd(self, reqId, start, end):
        print(reqId)
    # Get next order id we can use
    def nextValidId(self, nextorderId):
        global orderId
        orderId = nextorderId
    # Listen for realtime bars
    def realtimeBar(self, reqId, time, open_, high, low, close,volume, wap, count):
        super().realtimeBar(reqId, time, open_, high, low, close, volume, wap, count)
        try:
            bot.on_bar_update(reqId, time, open_, high, low, close, volume, wap, count)
        except Exception as e:
            print(e)
    def error(self, id, errorCode, errorMsg):
        print(errorCode)
        print(errorMsg)

#Bar Object
class Bar:
    open = 0
    low = 0
    high = 0
    close = 0
    volume = 0
    date = datetime.now()
    def __init__(self):
        self.open = 0
        self.low = 0
        self.high = 0
        self.close = 0
        self.volume = 0
        self.date = datetime.now()

#Bot Logic
class Bot:
    ib = None
    barsize = 5
    currentBar = Bar()
    bars = []
    reqId = 1
    global orderId
    symbol = "TSLA"
    speicher = []
    IBool = True
    initialbartime = datetime.now().astimezone(pytz.timezone("America/New_York"))
    lastOrderID = 0
    def __init__(self):
        #Connect to IB on init
        self.ib = IBApi()
        self.ib.connect(open(r"C:\Users\jensb\Documents\Praktikum\Stock_Prediction\03-07-2021\IBpasswords.txt", "r").readlines()[0], 7497, 1)
        ib_thread = threading.Thread(target=self.run_loop, daemon=True)
        ib_thread.start()
        time.sleep(1)
        currentBar = Bar()
        mintext = " min"
        if (int(self.barsize) > 1):
            mintext = " mins"
        queryTime = (datetime.now().astimezone(pytz.timezone("America/New_York"))-timedelta(days=1)).replace(hour=16,minute=0,second=0,microsecond=0).strftime("%Y%m%d %H:%M:%S")
        #Create our IB Contract Object
        contract = Contract()
        contract.symbol = self.symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        self.ib.reqIds(-1)
        # Request Market Data
        #self.ib.reqRealTimeBars(0, contract, 5, "TRADES", 1, [])
        self.ib.reqHistoricalData(self.reqId,contract,"","3 D",str(self.barsize)+mintext,"TRADES",1,1,True,[])
        #self.ib.reqHistoricalData(self.reqId, contract, "", )
        #Load all Models and Signals
        checkpoint = torch.load(r"C:\Users\jensb\Documents\Praktikum\Stock_Prediction\03-07-2021\JugendForscht\deepLearning\PricePredictionModel_hourly.pt")
        self.learning_rate = 0.0005
        self.inputs = 5
        self.hiddens = 60
        self.layers = 1
        self.outputs = 1
        self.seq_length = 5

        self.model = Netz(self.outputs, self.inputs, self.hiddens, self.layers, 5)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
        '''
        print("Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            print(var_name, "\t", self.optimizer.state_dict()[var_name])
        '''


    #Listen to socket in seperate thread
    def run_loop(self):
        self.ib.run()

    #Bracet Order Setup
    def bracketOrder(self, parentOrderId, action, quantity, profitTarget, stopLoss):
        #Initial Entry
        #Create our IB Contract Object
        contract = Contract()
        contract.symbol = self.symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"
        # Create Parent Order / Initial Entry
        parent = Order()
        parent.orderId = parentOrderId
        lastOrderID = parentOrderId
        parent.orderType = "MKT"
        parent.action = action
        parent.totalQuantity = quantity
        parent.transmit = False
        # Profit Target
        profitTargetOrder = Order()
        profitTargetOrder.orderId = parent.orderId+1
        profitTargetOrder.orderType = "LMT"
        profitTargetOrder.action = "SELL" if action == "BUY" else "BUY"
        profitTargetOrder.totalQuantity = quantity
        profitTargetOrder.lmtPrice = round(profitTarget,2)
        profitTargetOrder.parentId = parentOrderId
        profitTargetOrder.transmit = False
        # Stop Loss
        stopLossOrder = Order()
        stopLossOrder.orderId = parent.orderId+2
        stopLossOrder.orderType = "STP"
        stopLossOrder.action = "SELL" if action == "BUY" else "BUY"
        stopLossOrder.totalQuantity = quantity
        stopLossOrder.parentId = parentOrderId
        stopLossOrder.auxPrice = round(stopLoss,2)
        stopLossOrder.transmit = True

        bracketOrders = [parent, profitTargetOrder, stopLossOrder]
        return bracketOrders, lastOrderID

    #Pass realtime bar data back to our bot object
    def on_bar_update(self, reqId, bar,realtime):
        global orderId
        self.speicher.append(bar)
        #Historical Data to catch up
        if (realtime == False):
            self.bars.append(bar)
        else:
            bartime = datetime.strptime(bar.date,"%Y%m%d %H:%M:%S").astimezone(pytz.timezone("America/New_York"))
            minutes_diff = (bartime-self.initialbartime).total_seconds() / 60.0
            self.currentBar.date = bartime
            lastBar = self.bars[len(self.bars)-1]
            #On Bar Close
            if (minutes_diff > 0 and math.floor(minutes_diff) % self.barsize == 0):
                self.bars[-1] = self.speicher[-2]
                self.initialbartime = bartime
                
                opens = []
                highs = []
                lows =[]
                closes = []
                
                for entry in self.bars:
                    opens.append(entry.open)
                    highs.append(entry.high)
                    lows.append(entry.low)
                    closes.append(entry.close)
                self.open_array = pd.Series(np.asarray(opens))
                self.high_array = pd.Series(np.asarray(highs))
                self.low_array = pd.Series(np.asarray(lows))
                self.close_array = pd.Series(np.asarray(closes))

                #Create realtime Bar
                thisOpen = opens[-1]
                thisLow = lows[-1]
                thisHigh = highs[-1]
                thisClose = closes[-1]

                #Create previous Bar
                prevOpen = opens[-2]
                prevLow = lows[-2]
                prevHigh = highs[-2]
                prevClose = closes[-2]

                #Print realtime/this Bar and previous Bar
                '''
                print(" ")
                print("This Bar: ", "Open:", thisOpen, " High: ", thisHigh, " Low:", thisLow, " Close:",thisClose)
                print("Previous Bar: ", "Open:", prevOpen, " High: ", prevHigh, " Low:", prevLow, " Close:",prevClose)
                '''
                
                # Check Buy Criteria
               

                #Ask Models

                #For the Order add the Indicators

                #Nicht Ändern Sonst funktioniert das system nicht mehr
                #Bar closed append
                self.currentBar.close = bar.close
                self.bars.append(self.currentBar)
                self.currentBar = Bar()
                self.currentBar.open = bar.open
                
        #Build  realtime bar
        if (self.currentBar.open == 0):
            self.currentBar.open = bar.open
        if (self.currentBar.high == 0 or bar.high > self.currentBar.high):
            self.currentBar.high = bar.high
        if (self.currentBar.low == 0 or bar.low < self.currentBar.low):
            self.currentBar.low = bar.low

#Start Bot
bot = Bot()