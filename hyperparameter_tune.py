#!/usr/bin/python3.6 hyperparameter_tune.py

import math
import numpy as np
import json
import pandas as pd
import sys
from pprint import pprint
import collections
from operator import itemgetter

def main():
	maxk = int(sys.argv[1])
	train_file = open(sys.argv[2])
	val_file = open(sys.argv[3])
	test_file = open(sys.argv[4])

	train_set = json.load(train_file)
	test_set = json.load(test_file)
	val_set = json.load(val_file)


	train_df = pd.DataFrame(train_set['data'])
	test_df = pd.DataFrame(test_set['data'])
	val_df = pd.DataFrame(val_set['data'])

	metadata_df = pd.DataFrame(train_set['metadata'])
	labels = metadata_df['features']
	labels1 = labels[len(labels) - 1]
	dictOfLabels = collections.OrderedDict()
	for la in labels1[1]:
		dictOfLabels[la] = 0

	metadata_numpy = np.array(train_set['metadata']['features'])
	train_numpy = np.array(train_set['data'])
	train_numpy = np.transpose(train_numpy)
	test_numpy = np.array(val_set['data'])
	test_numpy = np.transpose(test_numpy)
	number_numpy_test = []
	number_numpy_train = []
	word_numpy_test = []
	word_numpy_train = []
	index = 0

	for col_feat in metadata_numpy: 
		if col_feat[1] == 'numeric':
			number_numpy_train.append(train_numpy[index])
			number_numpy_test.append(test_numpy[index])
		elif col_feat[1] != 'numeric' and col_feat[0] != 'label':
			word_numpy_train.append(train_numpy[index])
			word_numpy_test.append(test_numpy[index])
		if (col_feat[0] == 'label'):
			number_numpy_train.append(train_numpy[index])
			number_numpy_test.append(test_numpy[index])
			word_numpy_train.append(train_numpy[index])
			word_numpy_test.append( test_numpy[index])
		index += 1 
	train_df_num = pd.DataFrame(np.transpose(number_numpy_train))
	test_df_num = pd.DataFrame(np.transpose(number_numpy_test))
	train_df_word = pd.DataFrame(np.transpose(word_numpy_train))
	test_df_word = pd.DataFrame(np.transpose(word_numpy_test))

	train_mean = train_df_num.loc[:,train_df_num.columns != len(train_df_num.columns)-1].mean()
	train_std =  train_df_num.loc[:,train_df_num.columns != len(train_df_num.columns)-1].std(ddof = 0)
	train_std = train_std.replace(0, 1)
	num_train = train_df_num
	num_test = test_df_num


	words_train = train_df_word
	words_test = test_df_word

#validator is the CORRECT labels of the test file
	validator = np.array(val_df.iloc[:,-1])
	
	dist_dict = collections.OrderedDict()

	if (len(num_test.columns) > 1):
		stan_train = getStan(train_mean, train_std, num_train)
		stan_test = getStan(train_mean, train_std, num_test)

		np_tr = np.array(stan_train)
		np_test = np.array(stan_test)

		dist_dict = computeManhattan(np_tr, np_test)
		(stringtoprint, minimumk) = getMin(maxk, dist_dict, dictOfLabels, validator, 0)


	if (len(words_test.columns) > 1):
		word_tr = np.array(words_train)
		word_test = np.array(words_test)

		ham_dict = computeHamming(word_tr, word_test, dist_dict)
		(stringtoprint, minimumk) = getMin(maxk, ham_dict, dictOfLabels, validator, 0)
	print(*stringtoprint, sep ='\n')
	print(minimumk)
########################################################RETRAIN FEATURE ON VALIDATION AND TRAIN DATA
####TEST ON TEST.JSON FILES
	trainval_df =  train_df.append(val_df)
	metadata_numpy = np.array(train_set['metadata']['features'])
	train_numpy = np.array(trainval_df)
	train_numpy = np.transpose(train_numpy)
	test_numpy = np.array(test_set['data'])
	test_numpy = np.transpose(test_numpy)
	number_numpy_test = []
	number_numpy_train = []
	word_numpy_test = []
	word_numpy_train = []
	index = 0
	for col_feat in metadata_numpy: 
		if col_feat[1] == 'numeric':
			number_numpy_train.append(train_numpy[index])
			number_numpy_test.append(test_numpy[index])
		elif col_feat[1] != 'numeric' and col_feat[0] != 'label':
			word_numpy_train.append(train_numpy[index])
			word_numpy_test.append(test_numpy[index])
		if (col_feat[0] == 'label'):
			number_numpy_train.append(train_numpy[index])
			number_numpy_test.append(test_numpy[index])
			word_numpy_train.append(train_numpy[index])
			word_numpy_test.append( test_numpy[index])
		index += 1 
	train_df_num = pd.DataFrame(np.transpose(number_numpy_train))
	test_df_num = pd.DataFrame(np.transpose(number_numpy_test))
	
	train_df_word = pd.DataFrame(np.transpose(word_numpy_train))
	test_df_word = pd.DataFrame(np.transpose(word_numpy_test))

	train_mean = train_df_num.loc[:,train_df_num.columns != len(train_df_num.columns)-1].mean()
	train_std =  train_df_num.loc[:,train_df_num.columns != len(train_df_num.columns)-1].std(ddof = 0)
	train_std = train_std.replace(0, 1)
	num_train = train_df_num
	num_test = test_df_num


	words_train = train_df_word
	words_test = test_df_word

	validator = np.array(test_df.iloc[:,-1])
	
	dist_dict = collections.OrderedDict()


	if (len(num_test.columns) > 1):
		stan_train = getStan(train_mean, train_std, num_train)
		stan_test = getStan(train_mean, train_std, num_test)

		np_tr = np.array(stan_train)
		np_test = np.array(stan_test)

		dist_dict = computeManhattan(np_tr, np_test)
		stringtoprint = getMin(minimumk, dist_dict, dictOfLabels, validator, 1)


	if (len(words_test.columns) > 1):
		word_tr = np.array(words_train)
		word_test = np.array(words_test)

		ham_dict = computeHamming(word_tr, word_test, dist_dict)
		stringtoprint = getMin(minimumk, ham_dict, dictOfLabels, validator, 1)

	print(*stringtoprint)


def nansumwrapper(a, **kwargs):
    if np.isnan(a).all():
        return np.nan
    else:
        return np.nansum(a, **kwargs)

#SET TWO OPTIONS FOR GETTING THE NEAREST NEIGHBORS. FINAL = 1 WHEN THE MINIMUM K IS FOUND
def getMin(maxk, distance_dict, labels, validator, final):
	listOfK = []
	stringarray = []
	if final == 1:
		k = maxk
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
	else:
		for k in range(1,maxk + 1):
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
			listOfK.append([k, percentageCorrect])
			stringarray.append(str(k) + ',' + str(percentageCorrect))

		sortedlist = sorted(listOfK, key=itemgetter(1), reverse=True)
		return (stringarray, sortedlist[0][0])
			


#SAME METHODS AS KNN_CLASSIFIER.PY
def getStan(dfmean, dfstd, df):
	stan_df = (df- dfmean)/dfstd
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
	return dict_dist

def computeHamming(words_train, words_test, ham_dict):
	ham_dict = collections.OrderedDict()
	instance1 = 0
	for row in words_test:
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