# -*- coding: utf-8 -*-

import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')  # has to be imported before pyplot
import matplotlib.pyplot as plt

from Configuration import Configuration

matplotlib.rc('font', family='Arial')
import numpy as np
import sys

results_dir = sys.argv[1]


def round_to_2_decimal(value):
    return round(value * 100, 2)


def main():
    print "Plot generator 0.1"

    directory = results_dir
    set_name = 'digits'
    x_nums = Configuration.SAMPLES_N
    # x_nums = ['300', '600', '900', '1200', '1500']

    for x_num in x_nums:
        # plot_and_save_all_ml_methods(directory, set_name, x_num, 'PCA', '.pdf')
        plot_and_save_all_ml_methods(directory, set_name, x_num, 'LinearDiscriminantAnalysis', '.pdf')

   
def plot_and_save_all_ml_methods(directory, set_name='digits', x_num='1500', reduction_method='PCA',
                                 format='.pdf'):
    ml_methods = ['ann', 'svm', 'forest', 'tree']
    colors = ['r', 'g', 'b', 'y']

    fig = plt.figure()

    fig, ax = plt.subplots()

    fig.set_size_inches(8, 6)

    for ml_method, color in zip(ml_methods, colors):
        data_path = construct_path(directory, set_name, x_num, ml_method, reduction_method)
        plot_from_file(ax, ml_method, data_path, reduction_method, x_num, color)

    plt.savefig(construct_path(results_dir + 'plots/', set_name, x_num, 'all', reduction_method, '_', format))


def construct_path(directory, set_name='digits', x_num='1500', ml_method='svm', reduction_method='PCA', sep='_',
                   suffix='.dat'):
    return directory + set_name + sep + str(x_num) + sep + ml_method + sep + reduction_method + suffix


i = 0

def print_rounded(feature_nums, scores, stds, ml_method, x_num):
    
    for score, std, feature_num in zip(scores, stds, feature_nums):
            print "{0} {1} {2} {3} {4}".format(x_num, ml_method, feature_num, round_to_2_decimal(score), round_to_2_decimal(std))




def plot_from_file(ax, ml_method, file_name, reduction_method, x_num, color='b'):
    data = read_data(file_name)
    feature_nums, scores, stds = prepare_data(data)

    scores, stds = np.array(scores), np.array(stds)

    print_rounded(feature_nums, scores, stds, ml_method, x_num)

    ax.plot(feature_nums, scores, color=color, label=ml_method)

    # ax.set_xscale('log')

    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    # ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 30, 64])

    ax.fill_between(feature_nums, scores + stds, scores - stds, alpha=0.2, color=color)

    if reduction_method == 'PCA':
        plt.title(u'Zależność skuteczności od liczby atrybutów dla PCA')
    else:
        plt.title(u'Zależność skuteczności od liczby atrybutów dla LDA')

    plt.ylabel(u'skuteczność +/- odchyelenie standardowe')
    plt.xlabel(u'liczba atrybutów')

    
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

    assert (len(feature_nums) == data_len and len(scores) == data_len and len(stds) == data_len)

    b1 = feature_nums[0]
    b2 = scores[0]
    b3 = stds[0]
    return feature_nums[::-1], scores[::-1], stds[::-1]


def floats_from_str(str):
    l = []

    for t in str.split():
        try:
            l.append(float(t))
        except ValueError:
            pass

    return l[0], l[1], l[2]


if __name__ == '__main__':
    main()
