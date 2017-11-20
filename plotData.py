import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_decision_regions(X_train, X_test, y_train, y_test, clf):

    X = np.append(X_train, X_test, axis=0)
    y = np.append(y_train, y_test, axis=0)
    
    n_classes = len(np.unique(y))
    
    xx, yy = make_meshgrid(X, y, h=0.01)
    
    model = clf.fit(X_train, y_train)

    fig = plt.figure()
    axs = plt.gca()
    
    plot_contours(axs, model, xx, yy)

    # plot training data with 'x's
    plot_data(X_train, y_train, axs, 'o')
    # plot unknown data with 'o's
    plot_data(X_test, y_test, axs, 'x')

    plt.show()
    return

def make_meshgrid(x, y, h=.02):

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, model, xx, yy, **params):
 
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap = plt.cm.Set3, alpha=0.5, **params)
    return out

def plot_data(X, y, axs, m='x'):

    n_classes = len(np.unique(y))
    colors = ['blue', 'red', 'green']
    # plot data with colors according to class labels
    for l, c in zip(range(n_classes), colors):
        xs = []
        for xi, yi in zip(X, y):
            if yi == l:
                xs.append(xi)
        xs = np.array(xs)
        axs.scatter(xs[:, 0], xs[:, 1], color=c, marker=m, alpha=1.0, edgecolor='black')
    return

def plot_KMeans(X, clf):
    
    plt.figure(figsize=(10,7))
    colormap = np.array(['red', 'lime', 'black'])
    model = clf.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=colormap[clf.labels_], s=40)
    plt.show()

