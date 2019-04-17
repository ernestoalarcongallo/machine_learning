import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    """ The Perceptron classifier
    PARAMETERS:
    -----------
    eta: float, a learning rate
    n_iter: int, the number of iterations
    random_state: int, a random seed to initialize the weights

    ATTRIBUTES:
    ----------
    w_: 1D array -> weights after fitting
    errors: list -> the errors after each iteration
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(X, y):
        """ The fit method for training data
        Parameters:
        -----------
        X: {array-like}, shape: [n_samples, n_features]  example np.array([[1, 2, 3], [4, 5, 6]])
        y: array, shape: [n_samples] example [1, -1]

        Returns:
        --------
        self: object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.01, scale=0.01, size=1 + X.shape[1]) # bias + shape

        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi)) # 1 - (-1)
                self.w_[1:] += update * xi
                self.w_[0] += update # w0*x0 = Bias*1
                errors += int(update != 0.0)
        
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w[1:]) + self.w_[0] # wT*X

    def predict(self, X):
        """Calculate the class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

# Select setosa and versicolor
# |Sepal lenght|Sepal widht|Petal lenght|Petal widht|
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# Select sepal and petal lenght
X = df.iloc[0:100, [0, 2]].values

# Plot them
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('sepal lenght [cm]')
plt.ylabel('petal lenght [cm]')
plt.legend(loc='uppe left')
plt.show()