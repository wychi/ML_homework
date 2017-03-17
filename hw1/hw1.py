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

	train(max_iteration, init_learning_rate, gradientStop, verbose)
	

def test(testing, w):
	print '====test======'
	print 'test w=', w
	for i in range(5):
		print 'expected=', testing['PM2.5'][i]
		test = testing.iloc[i,:]
		print 'answer= ', w.dot(test)
	print '  '

def train(max_iteration, init_learning_rate, gradientStop, verbose):
	df = pd.read_csv('./data/train.csv', na_values='NR', encoding='big5')
	days = df[u'日期'].unique()[:5]
	# delete('RAINFALL')
	factors = df[u'測項'].unique()
	factors = np.delete(factors, 10)

	# order matters
	# days = ['2014/1/1']
	#factors = [u'PM10', u'PM2.5']
	factors = [u'PM10', u'PM2.5', u'WIND_DIREC', u'WIND_SPEED']

	print factors
	print days

	frames = []
	for day in days:
		dfByDay = df[df[u'日期'].isin([day]) & df[u'測項'].isin(factors)].iloc[:, 3:]
		dfByDay.index = factors
		frames.append(dfByDay)
	training = pd.concat(frames, axis = 1)
	training = training.T
	training['b'] = 1

	x = training[:-1]
	y = training['PM2.5'][1:]

	# [w1, w2, w3, ..., b]
	#w = np.random.rand( len(factors) + 1)
	#w= np.array([  0.70403291, 11.17861785])
	#w = np.array([ 0.12668465  ,0.73373688  ,0.00877731  ,0.75455953 ,0.92578519])
	w = np.ones(len(factors) + 1)

	wStar = gradient_descent(w, x, y, max_iteration, init_learning_rate, gradientStop, verbose)
	validate(wStar, x, y)

def validate(w, x, y):
	error = np.subtract(y, x.dot(w))
	errorRate = error / y;
	print 'error rate', np.average(np.fabs(errorRate))

def gradient_descent(w, x, y, max_iteration, init_learning_rate, gradientStop, verbose):

	loop = 0
	rate = init_learning_rate

	print 'init w=', w
	print 'inti rate=', init_learning_rate
	print '---- start -----'

	L_history = []
	gradientChange = 1
	while True :
		loop = loop + 1
		error = np.subtract(y, x.dot(w))
		dw = np.dot(error, -x)
		gradientChange = np.dot(dw, dw)
		w = w - dw * rate
		#rate = init_rate
		L = np.sqrt(np.dot(error, error))
		
		if(verbose and loop % 10 == 0):
			errorRate = np.average(np.fabs(error / y))
			print 'iter #',loop,' Lost= ', L, 'error', errorRate, 'rate= ', rate, 'gradient', gradientChange
			L_history.append(L)

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

	# plt.plot(L_history[3:])
	# plt.show()

	return w


if __name__ == "__main__":
	main(sys.argv[1:])
