from AlgorithmImports import *

import datetime
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from deap import base
from deap import creator
from deap import tools
from tqdm import trange
from pathlib import Path


#with open("txtData.txt", "r") as f:
#    data = f.readlines()
#    slowV = int(data[0])
#    fastV = int(data[1])
#    toleranceV = float(data[2])
#    rsiSELL = int(data[3])

#Settings.slowV = 50
#Settings.fastV = 15
#Settings.toleranceV = 0.00015
#Settings.rsiSELL = 20


class MovingAverageCrossAlgorithm(QCAlgorithm):

    def Initialize(self):
        '''Initialise the data and resolution required, as well as the cash and start-end dates for your algorithm. All algorithms must initialized.'''

        slowV = int(self.GetParameter("movingAvgSlow"))
        fastV = int(self.GetParameter("movingAvgFast"))
        #rsiSELL = 20

        self.SetStartDate(2009, 1, 1)    #Set Start Date
        self.SetEndDate(2015, 1, 1)      #Set End Date
        self.SetCash(100000)             #Set Strategy Cash
        # Find more symbols here: http://quantconnect.com/data
        self.spy = self.AddEquity("SPY", Resolution.Daily)
        self.spy.SetDataNormalizationMode(DataNormalizationMode.Raw)
        # create a 15 day exponential moving average
        self.fast = self.EMA("SPY", fastV, Resolution.Daily)

        # create a 30 day exponential moving average
        self.slow = self.EMA("SPY", slowV, Resolution.Daily)
        
        #RSI
        self.rsi = self.RSI("SPY", 10,  MovingAverageType.Simple, Resolution.Daily)
        # initialize the indicator with the daily history close price
        history = self.History(["SPY"], 10, Resolution.Daily)
        for time, row in history.loc["SPY"].iterrows():
            self.rsi.Update(time, row["close"])

        self.previous = None
    def OnData(self, data):
        
        #slowV = self.GetParameter
        #fastV = 15
        rsiSELL = int(self.GetParameter("rSIselling"))

        '''OnData event is the primary entry point for your algorithm. Each new data point will be pumped in here.'''
        # a couple things to notice in this method:
        #  1. We never need to 'update' our indicators with the data, the engine takes care of this for us
        #  2. We can use indicators directly in math expressions
        #  3. We can easily plot many indicators at the same time

        # wait for our slow ema and RSI to fully initialize
        if not self.slow.IsReady:
            return
        if not self.rsi.IsReady: 
            return
        
        # only once per day
        if self.previous is not None and self.previous.date() == self.Time.date():
            self.Debug("**timeIssue**")
            return

        # define a small tolerance on our checks to avoid bouncing
        tolerance = 0.00015

        holdings = self.Portfolio["SPY"].Quantity

        # we only want to go long if we're currently short or flat
        if holdings <= 0:
            # if the fast is greater than the slow, we'll go long
            if self.fast.Current.Value > self.slow.Current.Value *(1 + tolerance):
                #self.Log("BUY  >> {0}".format(self.Securities["SPY"].Price))
                self.SetHoldings("SPY", 1.0)

        # we only want to liquidate if we're currently long
        # if the fast is less than the slow we'll liquidate our long
        if holdings > 0 and ((self.fast.Current.Value < self.slow.Current.Value) or self.rsi.Current.Value < rsiSELL):
            #self.Log("SELL >> {0}".format(self.Securities["SPY"].Price))
            self.Liquidate("SPY")

        self.previous = self.Time



