from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()
X, y = iris.data[:, :2], iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Define parameter values to test
C_values = [0.1, 1, 10]
gamma_values = [0.1, 1, 10]

# Open a file to write results
with open("svc_results.txt", "w") as f:
    f.write("C,gamma,accuracy\n")

    # Loop over parameter values and train a model for each combination
    for C in C_values:
        for gamma in gamma_values:
            clf = SVC(C=C, gamma=gamma, random_state=42)
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test, y_test)

            # Write results to file
            f.write(f"{C},{gamma},{accuracy}\n")

            # Plot mesh for current model
            xx, yy = np.meshgrid(np.linspace(4, 8, 100),
                                 np.linspace(1.5, 4.5, 100))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, alpha=0.4)
            plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
            plt.title(
                f"C={C}, gamma={gamma}, accuracy={accuracy}")
            plt.show()
