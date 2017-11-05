# -*- coding: utf-8 -*-

import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')  # has to be imported before pyplot
import matplotlib.pyplot as plt

from Configuration import Configuration

matplotlib.rc('font', family='Arial')
import numpy as np
import sys
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

results_dir = sys.argv[1]


def round_to_2_decimal(value):
    return round(value, 2)


def main():
    print "Plot generator 0.1"

    directory = results_dir
    set_name = 'digits'
    x_nums = Configuration.SAMPLES_N


    figure = plt.figure()
    gs1 = gridspec.GridSpec(3, 2)
    figure.suptitle(u"Skuteczność modelu w zależności od liczby wykorzystanych atrybutów\n przy różnej liczbie rekordów użytych przy uczeniu", fontsize=18)

    figure.set_size_inches(12,16)
    
    i = 0
    for x_num in x_nums:
        # plot_and_save_all_ml_methods(directory, set_name, x_num, 'PCA', '.pdf')
        ax = figure.add_subplot(gs1[i])
        plot_and_save_all_ml_methods(ax, directory, set_name, x_num, 'LinearDiscriminantAnalysis', '.pdf')
        i += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.87])

    colors = ['r', 'g', 'b', 'y']

    l1 = Line2D([0, 1], [0, 1], transform=figure.transFigure, figure=figure, c=colors[0])
    l2 = Line2D([0, 1], [0, 1], transform=figure.transFigure, figure=figure, c=colors[1])
    l3 = Line2D([0, 1], [0, 1], transform=figure.transFigure, figure=figure, c=colors[2])
    l4 = Line2D([0, 1], [0, 1], transform=figure.transFigure, figure=figure, c=colors[3])

    l5 = Line2D([0, 1], [0, 1], transform=figure.transFigure, figure=figure, c=colors[0], linestyle='--')
    l6 = Line2D([0, 1], [0, 1], transform=figure.transFigure, figure=figure, c=colors[1], linestyle='--')
    l7 = Line2D([0, 1], [0, 1], transform=figure.transFigure, figure=figure, c=colors[2], linestyle='--')
    l8 = Line2D([0, 1], [0, 1], transform=figure.transFigure, figure=figure, c=colors[3], linestyle='--')

    figure.legend((l1, l5, l2, l6, l3, l7, l4, l8), ('ann', u'best ann', 'svm', u'best svm', 'forest', u'best forest', 'tree', u'best tree'), (0.06, 0.875), ncol=4)

    plt.savefig("summaryExtraction.png")

def to_percent(array):
    for idx, val in enumerate(array):
        array[idx] = val*100


def plot_and_save_all_ml_methods(ax, directory, set_name='digits', x_num='1500', reduction_method='PCA',
                                 format='.pdf'):
    ml_methods = ['ann', 'svm', 'forest', 'tree']
    colors = ['r', 'g', 'b', 'y']

    for ml_method, color in zip(ml_methods, colors):
        data_path = construct_path(directory, set_name, x_num, ml_method, reduction_method)
        plot_from_file(ax, ml_method, data_path, reduction_method, x_num, color)

def construct_path(directory, set_name='digits', x_num='1500', ml_method='svm', reduction_method='PCA', sep='_',
                   suffix='.dat'):
    return directory + set_name + sep + str(x_num) + sep + ml_method + sep + reduction_method + suffix



def print_rounded(feature_nums, scores, stds, ml_method, x_num):
    
    for score, std, feature_num in zip(scores, stds, feature_nums):
            print "{0} {1} {2} {3} {4}".format(x_num, ml_method, feature_num, round_to_2_decimal(score), round_to_2_decimal(std))




def plot_from_file(ax, ml_method, file_name, reduction_method, x_num, color='b'):
    data = read_data(file_name)
    feature_nums, scores, stds = prepare_data(data)

    to_percent(scores)
    to_percent(stds)

    scores, stds = np.array(scores), np.array(stds)

    print_rounded(feature_nums, scores, stds, ml_method, x_num)

    ax.plot(feature_nums, scores, color=color, label=ml_method)

    # ax.set_xscale('log')

    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xticks([1, 2, 3, 4])

    ax.fill_between(feature_nums, scores + stds, scores - stds, alpha=0.2, color=color)

    plt.title(u"{0} rekordów".format(x_num))

    plt.ylabel(u'skuteczność +/- odchyelenie standardowe (%)')
    plt.xlabel(u'liczba atrybutów')

    
    ax.axhline(np.max(scores), linestyle='--', color=color)

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
