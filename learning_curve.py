#!/usr/bin/python3.6 learning_curve.py

import math
import numpy as np
import json
import pandas as pd
import sys
from pprint import pprint
import collections
from operator import itemgetter

def main():
	k = int(sys.argv[1])
	train_file = open(sys.argv[2])
	test_file = open(sys.argv[3])

	train_set = json.load(train_file)
	test_set = json.load(test_file)

	#sys.stdout = open("out.txt" , "w")

	train_df_original = pd.DataFrame(train_set['data'])
	test_df = pd.DataFrame(test_set['data'])

	metadata_df = pd.DataFrame(train_set['metadata'])
	labels = metadata_df['features']
	labels1 = labels[len(labels) - 1]
	#print(labels1[1])
	dictOfLabels = collections.OrderedDict()
	for la in labels1[1]:
		dictOfLabels[la] = 0
	#filler, gotta change this later
	total_rows = train_df_original.shape[0]
	dist_dict = collections.OrderedDict()
	for cut in range(1,11):
		percent = cut*.1
		rowsToKeep = math.floor(percent*total_rows)
		train_df = train_df_original.head(rowsToKeep)
		train_mean = train_df.loc[:,train_df.columns != len(train_df.columns)-1].select_dtypes(include=[np.number]).mean()
		train_std =  train_df.loc[:,train_df.columns != len(train_df.columns)-1].select_dtypes(include=[np.number]).std(ddof = 0)
		train_std = train_std.replace(0, 1)

		num_train = train_df.select_dtypes(include=[np.number])
		num_test = test_df.select_dtypes(include=[np.number])

		words_train = train_df.select_dtypes(exclude=[np.number])
		words_test = test_df.select_dtypes(exclude=[np.number])

		validator = np.array(test_df.iloc[:,-1])
		if (len(num_test.columns) > 0):
			stan_train = getStan(train_mean, train_std, num_train)
			stan_test = getStan(train_mean, train_std, num_test)

			np_tr = np.array(stan_train)
			np_test = np.array(stan_test)

			dist_dict = computeManhattan(np_tr, np_test)
		#	print(dist_dict)
			stringtoprint = getMin(k, dist_dict, dictOfLabels, validator)
		#	print(dist_dict)
			####RETRAIN


		if (len(words_test.columns) > 0):
			word_tr = np.array(words_train)
			word_test = np.array(words_test)

			ham_dict = computeHamming(word_tr, word_test, dist_dict)
			stringtoprint = getMin(k, ham_dict, dictOfLabels, validator)
			####RETRAIN
		print(str(rowsToKeep) + ','+ str(*stringtoprint), sep ='\n')

def nansumwrapper(a, **kwargs):
    if np.isnan(a).all():
        return np.nan
    else:
        return np.nansum(a, **kwargs)


def getMin(k, distance_dict, labels, validator):
	stringarray = []
	total = 0
	correct = 0
	for i in distance_dict:
		total += 1
		sortedguy = sorted(distance_dict[i], key=lambda tup:tup[1])
		firstk = sortedguy[0:k]
		d = labels.fromkeys(labels, 0)
		for tuples in firstk:
			if tuples[2] in d:
				d[tuples[2]] += 1
		maximum = max(d, key=d.get)
		if maximum == validator[i - 1]:
			correct += 1
	percentageCorrect = correct/total
	stringarray.append(percentageCorrect)
	return stringarray



def getStan(dfmean, dfstd, df):
	stan_df = (df- dfmean)/dfstd
	#stan_df[len(stan_df.columns)] = df.iloc[:,-1]
	stan_df[len(stan_df.columns) - 1] = df.iloc[:,-1]
	return stan_df


def computeManhattan(stan_train, stan_test):
	dict_dist = collections.OrderedDict()
	instance1 = 0
	for row in stan_test:
		instance1 += 1
		list_of_instances = []
		dict_dist[instance1] = list_of_instances
		instance = 0
		rowtosub = row[0:len(row) - 1]
		for row2 in stan_train:
			instance += 1
			label = row2[-1]
			iterrow = row2[0:len(row2) - 1]
			newone = np.absolute(np.subtract(rowtosub, iterrow))
			distance = np.sum(newone)
			dict_dist[instance1].append((instance, distance, label))	
	#print(instance1)
	return dict_dist

def computeHamming(words_train, words_test, ham_dict):
	ham_dict = collections.OrderedDict()
	instance1 = 0
	for row in words_test:
	#	print(row)
		instance1 += 1
		list_of_instances = []
		ham_dict[instance1] = list_of_instances
		instance = 0
		rowtosub = row[:len(row) - 1]
		for row2 in words_train:
			instance += 1
			label = row2[-1]
			iterrow = row2[:len(row2) - 1]
			newArray = np.equal(iterrow, rowtosub)
			distance = np.size(newArray) - np.count_nonzero(newArray)
			instance_append = ((instance, distance, label))
			ham_dict[instance1].append(instance_append)

	return ham_dict

if __name__ == '__main__':
	main() 