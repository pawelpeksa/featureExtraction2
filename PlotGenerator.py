 # -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib
import numpy as np



def main():
	print "Plot generator 0.1"

	matplotlib.rc('font', family='Arial')

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
	
	plt.figure().set_size_inches(8, 7)
	plt.title(u'Zależność skuteczności od wymiaru danych\n przy zastosowaniu metody ' + reduction_method + u'\ndla ilości danych N=' + x_num)

	for ml_method, color in zip(ml_methods, colors):
		data_path = construct_path(directory, set_name, x_num, ml_method, reduction_method)
		plot_from_file(data_path, color, ml_method)

	plt.xticks(np.arange(0, 64, 5))

	plt.legend(framealpha=0.5, loc='best')
	plt.savefig(construct_path('./results0/plots/', set_name, x_num, 'all', reduction_method, '_', format))

def construct_path(directory='./results0/', set_name = 'digits', x_num='1500', ml_method = 'svm', reduction_method='PCA', sep = '_', suffix = '.dat'):
	return directory + set_name + sep + x_num + sep + ml_method + sep + reduction_method + suffix

def plot_from_file(file_name, color = 'b', label = 'label'):
	data = read_data(file_name)
	feature_nums, scores, stds = prepare_data(data)
	

	scores, stds = np.array(scores), np.array(stds)

	plt.plot(feature_nums, scores, color = color, label=label)

	plt.fill_between(feature_nums, scores + stds, scores - stds, alpha=0.2, color = color)

	plt.ylabel(u'Skuteczność +/- odchylenie standardowe')
	plt.xlabel('Wymiar danych')

	plt.axhline(np.max(scores), linestyle='--', color=color)

	plt.xlim([feature_nums[0], feature_nums[-1]])
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

