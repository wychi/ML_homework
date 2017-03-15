#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd

df = pd.read_csv('./data/train.csv', na_values='NR', encoding='big5')

days = df[u'日期'].unique()[:5]
# delete('RAINFALL')
factors = df[u'測項'].unique()
factors = np.delete(factors, 10)

# order matters
# days = ['2014/1/1']
factors = [u'PM10', u'PM2.5']
#factors = [u'PM10', u'PM2.5', u'WIND_DIREC', u'WIND_SPEED']

frames = []
for day in days:
	dfByDay = df[df[u'日期'].isin([day]) & df[u'測項'].isin(factors)].iloc[:, 3:]
	dfByDay.index = factors
	frames.append(dfByDay)

training = pd.concat(frames, axis = 1)
training = training.T
training['b'] = 1

testing = training.iloc[-5:,:]
training = training.iloc[:-5,:]

def test(w):
	print '====test======'
	print 'test w=', w
	for i in range(5):
		print 'expected=', testing['PM2.5'][i]
		test = testing.iloc[i,:]
		print 'answer= ', w.dot(test)
	print '  '

loop = 0
init_rate = 0.000001
rate = init_rate
# [w1, w2, w3, b]
w = np.ones(len(factors) + 1)
#w = np.random.rand( len(factors) + 1)
#w= np.array([  0.70403291, 11.17861785])
#w = np.array([ 0.12668465  ,0.73373688  ,0.00877731  ,0.75455953 ,0.92578519])

print factors
print days
print 'w=', w
print 'rate=', rate
print '---- start -----'

test(w)

while True :
	loop = loop + 1

	i = 0
	delta = np.array([])
	for idx, row in training.iloc[:-1,:].iterrows():
		x = row
		i = i + 1
		y = training['PM2.5'][i]
		dy = y - w.dot(x)
		delta = np.append(delta, dy)
	dw = delta.dot(-training.iloc[:-1,:])

	# update
	w = w - dw * rate
	rate = init_rate/np.sqrt(loop+1)

	L = delta.sum()
	print 'iter #',loop,' Lost= ', L, 'rate= ', rate

	if(loop % 1000 == 0):
		print 'Lost= ', L
		print 'rate= ', rate
		print w
		print '----', loop, '-----'

	if(abs(L) < 0.01):
		break

print '---- end -----'
test(w)
