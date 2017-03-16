#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./data/train.csv', na_values='NR', encoding='big5')

days = ['2014/1/1', '2014/1/2', '2014/1/3']
factors = df[u'測項'].unique()

frames = []
for day in days:
	dfByDay = df[df[u'日期'].isin([day])].iloc[:, 3:]
	dfByDay.index = factors
	frames.append(dfByDay)

training = pd.concat(frames, axis = 1)
training = training.T
training.index = pd.DatetimeIndex(start=days[0], freq='H', periods=len(days)*24)

ax = training.plot(linestyle='', marker='o', y='PM2.5', label='PM2.5')
training.plot(linestyle='', marker='o', ax = ax, y='PM10', label='PM10')

training.plot(kind="scatter", x="PM2.5", y="PM10")
training.plot(kind="scatter", x="PM2.5", y="WIND_DIREC")


plotNum = 1
for factor in factors:
	ax = plt.subplot(4,5, plotNum)
	plotNum += 1
	training.plot(kind="scatter", x="PM2.5", y=factor, ax =ax)

plt.show()
