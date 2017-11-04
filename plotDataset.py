from __future__ import print_function

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg') # has to be imported before pyplot


iris = datasets.load_iris()
x = iris.data
y = iris.target

lda = LinearDiscriminantAnalysis(n_components=2)
lda = lda.fit(x,y)
x_2 = lda.transform(x)

lda = LinearDiscriminantAnalysis(n_components=1)
lda = lda.fit(x,y)
x_1 = lda.transform(x)


colors = ['g', 'c', 'b']

figure = plt.figure()

figure.set_size_inches(14, 10)

gs1 = gridspec.GridSpec(3, 1)

ax1 = figure.add_subplot(gs1[0])
ax2 = figure.add_subplot(gs1[1])

for xn, yn in zip(x_1, y):
    ax1.scatter(xn[0], -50, color=colors[yn])

for xn, yn in zip(x_2, y):
    ax2.scatter(xn[0], xn[1], color=colors[yn])

plt.savefig('./testplot.png')



    