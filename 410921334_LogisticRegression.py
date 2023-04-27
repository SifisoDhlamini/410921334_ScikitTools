import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

# load iris dataset
iris = datasets.load_iris()

# split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    iris.data[:, :2], iris.target, test_size=0.2, random_state=42)

# define a list of models and their hyperparameters
models = [{'name': 'Logistic Regression with L1 regularization', 'model': LogisticRegression(penalty='l1', solver='saga', C=0.01)},    {
    'name': 'Logistic Regression with L2 regularization', 'model':
    LogisticRegression(penalty='l2', solver='saga', C=0.01)}]

# train and evaluate each model
for model in models:
    algorithm_name = model['name']
    params = model['model'].get_params()
    print('Training', algorithm_name, 'with parameters', params)

    # train the model
    model['model'].fit(X_train, y_train)

    # predict on test data
    y_pred = model['model'].predict(X_test)

    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

    # save results to a file
    with open('results.txt', 'a') as f:
        f.write('Algorithm: {}\n'.format(algorithm_name))
        f.write('Parameters: {}\n'.format(params))
        f.write('Accuracy: {}\n\n'.format(accuracy))

    # plot decision boundary
    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model['model'].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(algorithm_name)

plt.show()
