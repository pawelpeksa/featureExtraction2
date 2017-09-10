# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Arial')
import numpy as np


def main():
	print "Plot generator 0.1"

	directory = './results0/' 
	set_name = 'digits'
	x_nums = ['300', '600']
	# x_nums = ['300', '600', '900', '1200', '1500']

	for x_num in x_nums:
		plot_and_save_all_ml_methods(directory, set_name, x_num, 'PCA', '.pdf')	
		plot_and_save_all_ml_methods(directory, set_name, x_num, 'LinearDiscriminantAnalysis', '.pdf')	

def plot_and_save_all_ml_methods(directory = './results0/', set_name = 'digits', x_num = '1500', reduction_method = 'PCA', format = '.pdf'):
	ml_methods = ['ann', 'svm', 'forest', 'tree']
	colors = ['r', 'g', 'b', 'y']
	
	fig = plt.figure()

	fig, ax = plt.subplots()

	fig.set_size_inches(8, 6)
	

	for ml_method, color in zip(ml_methods, colors):
		data_path = construct_path(directory, set_name, x_num, ml_method, reduction_method)
		plot_from_file(ax, ml_method, data_path, reduction_method, color)


	plt.savefig(construct_path('./results0/plots/', set_name, x_num, 'all', reduction_method, '_', format))

def construct_path(directory='./results0/', set_name = 'digits', x_num='1500', ml_method = 'svm', reduction_method='PCA', sep = '_', suffix = '.dat'):
	return directory + set_name + sep + x_num + sep + ml_method + sep + reduction_method + suffix

def plot_from_file(ax, ml_method, file_name, reduction_method, color = 'b'):
	data = read_data(file_name)
	feature_nums, scores, stds = prepare_data(data)
	

	scores, stds = np.array(scores), np.array(stds)

	ax.plot(feature_nums, scores, color = color, label=ml_method)

	ax.set_xscale('log')

	ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

	ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 30, 64])

	ax.fill_between(feature_nums, scores + stds, scores - stds, alpha=0.2, color = color)

	if reduction_method == 'PCA':
		plt.title(u'Zależność skuteczności od liczby atrybutów dla PCA')
	else:
		plt.title(u'Zależność skuteczności od liczby atrybutów dla LDA')

	plt.ylabel(u'skuteczność +/- odchyelenie standardowe')
	plt.xlabel(u'liczba atrybutów')

	bestScore = np.max(scores)
	bestIndex = scores.tolist().index(bestScore)

	print "Best for method: {0} is {1} for {2} features".format(ml_method, bestScore, feature_nums[bestIndex])
	print "Score for 64 features is: {0} is {1}".format(ml_method, scores[len(scores)-1])
	print ""

	ax.axhline(np.max(scores), linestyle='--', color=color)

	plt.xlim([feature_nums[0], feature_nums[-1]])
	plt.legend(loc="lower right", ncol=2)

	# plt.ylim([0, np.max(scores) + 0.01]) # worst score - its std goes under 0, should y lim be set to y= [0, smth] ?

def read_data(file_name):
	with open(file_name, 'r') as data_file:
		data = data_file.readlines()
	return data	

def prepare_data(data):	
	data = [x.strip() for x in data]
	
	feature_nums = list()
	scores = list()
	stds = list()

	for point in data:
		f_num, score, std = floats_from_str(point)

		feature_nums.append(f_num)
		scores.append(score)
		stds.append(std)

	data_len = len(data)

	assert(len(feature_nums) == data_len and len(scores) == data_len and len(stds) == data_len)

	return feature_nums, scores, stds

def floats_from_str(str):
	l = []

	for t in str.split():
		try:
			l.append(float(t))
		except ValueError:
			pass

	return l[0], l[1], l[2]    	

if __name__ == 	'__main__':
	main()

