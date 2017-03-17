#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys, getopt

def main(argv):
	try:
		opts, args = getopt.getopt(sys.argv[1:],"hi:r:v:",["iteration=","learning_rate=", "verbose="])
	except getopt.GetoptError:
		print 'test.py -i <iteration> -r <learning_rate>'
		sys.exit(2)

	max_iteration = 100000000
	init_learning_rate = 0.000001
	gradientStop = 0.01
	verbose = False

	for opt, arg in opts:
		if opt == '-h':
			print 'hw1.py -i <iteration> -o <learning_rate>'
			sys.exit()
		elif opt in ("-i", "--iteration"):
			max_iteration = int(arg)
		elif opt in ("-r", "--learning_rate"):
			init_learning_rate = float(arg)
		elif opt in ("-v", "--verbose"):
			verbose = True

	wStar = train(max_iteration, init_learning_rate, gradientStop, verbose)

def test(w, x, y):
	print '====test======'
	print 'test w=', w
	result = x.dot(w)
	for i in range(5):
		print 'expected=', y[i]
		print 'answer= ', result[i]
	print '  '

def train(max_iteration, init_learning_rate, gradientStop, verbose):
	df = pd.read_csv('./data/train.csv', na_values='NR', encoding='big5')

	# TODO n-ford
	days = df[u'日期'].unique()
	days = days[:days.shape[0] / 5]
	# delete('RAINFALL')
	factors = df[u'測項'].unique()
	factors = np.delete(factors, 10)

	# order matters
	#days = ['2014/1/1', '2014/1/2']
	#factors = [u'PM10', u'PM2.5']
	factors = [u'PM10', u'PM2.5', u'NO2']

	print factors
	print days

	frames = []
	for day in days:
		dfByDay = df[df[u'日期'].isin([day]) & df[u'測項'].isin(factors)].iloc[:, 3:]
		dfByDay.index = factors
		frames.append(dfByDay)
	training = pd.concat(frames, axis = 1)
	training = training.T

	# 取九小時資料排成一排
	m = training.shape[0]
	d = {}
	for i in range(0, m-8):
		x = training.iloc[i:i+9]
		x = x.T.values.flatten()
		d[i] = x
	x = pd.DataFrame(d).T
	x['b'] = 1
	y = training[8:]['PM2.5']

	print '#x', x.shape[0], '#y', y.shape[0]

	# TODO 
	# prepare testing data
	days = df[u'日期'].unique()
	days = days[-3:]
	frames = []
	for day in days:
		dfByDay = df[df[u'日期'].isin([day]) & df[u'測項'].isin(factors)].iloc[:, 3:]
		dfByDay.index = factors
		frames.append(dfByDay)
	testing = pd.concat(frames, axis = 1)
	testing = testing.T

	# 取九小時資料排成一排
	m = testing.shape[0]
	d = {}
	for i in range(0, m-8):
		px = testing.iloc[i:i+9]
		px = px.T.values.flatten()
		d[i] = px
	tx = pd.DataFrame(d).T
	tx['b'] = 1
	ty = testing[8:]['PM2.5']
	print '#tx', tx.shape[0], '#ty', ty.shape[0]

	# 取1小時資料排成一排
	# x = training[:-1]
	# y = training['PM2.5'][1:]

	# [w1, w2, w3, ..., b]
	#w = np.random.rand( len(factors) + 1)
	#w= np.array([  0.70403291, 11.17861785])
	#w = np.array([ 0.12668465  ,0.73373688  ,0.00877731  ,0.75455953 ,0.92578519])
	w = np.ones(len(factors)*9 + 1)
	wStar = gradient_descent(w, x, y, max_iteration, init_learning_rate, gradientStop, verbose)

	test(wStar, tx, ty)
	return wStar

def gradient_descent(w, x, y, max_iteration, init_learning_rate, gradientStop, verbose):

	loop = 0
	rate = init_learning_rate

	print 'init w=', w
	print 'inti rate=', init_learning_rate
	print '---- start -----'

	L_history = []
	Errors_history = []
	gradientChange = 1
	m = x.shape[0]
	
	# L = SUM[ ( w.X - y )^2 ]
	# to save "-1" operation
	while True :
		loop = loop + 1

		error = np.subtract(x.dot(w), y)
		dw = np.dot(error, x) / m
		gradientChange = np.dot(dw, dw)
		w = w - dw * rate
		#rate = init_rate
		L = np.sqrt(np.dot(error, error))
		L_history.append(L)

		if(verbose and loop % 100 == 0):
			er = np.divide(np.fabs(error), y)
			er = er[~np.isinf(er)]
			errorRate = np.average(er)
			print 'iter #',loop,' Lost= ', L, 'error', errorRate, 'rate= ', rate, 'gradient', gradientChange
			Errors_history.append(errorRate)

		if(loop > max_iteration):
			print 'stop. exceed max iteration'
			break
		if(gradientChange < gradientStop):
			print 'stop. gradientChange small enough', gradientChange
			break
		if(np.isnan(L) or np.isinf(L)):
			print 'stop inf or nan'
			break

	print '---- end -----'

	for i in xrange(len(L_history)):
		if L_history[i] > 3000:
			L_history[i] = 3000

	plt.figure(1)
	plt.plot(L_history)
	plt.savefig('loss.png')
	plt.figure(2)
	plt.plot(Errors_history)
	plt.savefig('error_rate.png')

	return w


if __name__ == "__main__":
	main(sys.argv[1:])
