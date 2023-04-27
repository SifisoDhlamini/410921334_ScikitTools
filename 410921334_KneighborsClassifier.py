import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from matplotlib.colors import ListedColormap

iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

# Create color maps for 3 classes
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# Define the range of values for each parameter
n_neighbors_range = [1, 10, 19]
weights_options = ['uniform', 'distance']

# Loop over all parameter combinations
f = open("knn_results.txt", "w")
for weights in weights_options:
    for n_neighbors in n_neighbors_range:

        # Create an instance of Neighbours Classifier and fit the data.
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
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
        plt.title("3-Class classification (k = %i, weights = '%s')"
                  % (n_neighbors, weights))

        # Compute and print the accuracy score
        score = clf.score(X, Y)
        print("n_neighbors = %i, weights = '%s', accuracy = %.3f" %
              (n_neighbors, weights, score))
        f.write("n_neighbors = %i, weights = '%s', accuracy = %.3f\n" %
                (n_neighbors, weights, score))

f.close()
plt.show()
