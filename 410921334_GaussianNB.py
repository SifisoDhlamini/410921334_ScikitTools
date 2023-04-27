import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

# Create color maps for 3 classes
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Define the range of values for each parameter
var_smoothing_range = [1e-9, 1e-7, 1e-5]
priors_options = [None, [0.2, 0.5, 0.3]]

# Loop over all parameter combinations
f = open("gnb_results.txt", "w")
for prior in priors_options:
    for var_smoothing in var_smoothing_range:

        # Create an instance of Gaussian Naive Bayes Classifier and fit the data.
        clf = GaussianNB(var_smoothing=var_smoothing, priors=prior)
        clf.fit(X, Y)

        # Create a mesh of points that covers the full range of the data
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        # Use the classifier to predict the class of each mesh point
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Reshape the predicted class labels into a grid
        Z = Z.reshape(xx.shape)

        # Plot the mesh and the data points
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap_bold, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("3-Class classification (var_smoothing = %.9f, priors = %s)"
                  % (var_smoothing, prior))

        # Compute and print the accuracy score
        score = clf.score(X, Y)
        print("var_smoothing = %.9f, priors = %s, accuracy = %.3f" %
              (var_smoothing, prior, score))
        f.write("var_smoothing = %.9f, priors = %s, accuracy = %.3f\n" %
                (var_smoothing, prior, score))

f.close()
plt.show()
