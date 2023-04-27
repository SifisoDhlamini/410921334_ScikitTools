from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
# Only use the first two features for visualization purposes
X = iris.data[:, :2]
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define parameter values to test
n_estimators_values = [10, 50]
max_depth_values = [5, 10]

# Open a file to write results
with open("rf_results.txt", "w") as f:
    f.write("n_estimators,max_depth,accuracy\n")

    # Loop over parameter values and train a model for each combination
    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:
            clf = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)

            # Write results to file
            f.write(f"{n_estimators},{max_depth},{accuracy}\n")

            # Plot mesh for current model
            xx, yy = np.meshgrid(np.linspace(4, 8, 100),
                                 np.linspace(2, 5, 100))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.4)
            plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
            plt.title(
                f"n_estimators={n_estimators}, max_depth={max_depth}, accuracy={accuracy}")
            plt.show()
