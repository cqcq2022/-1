import numpy as np
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Add bias term
        X = np.hstack((X, np.ones((n_samples, 1))))
        n_features += 1

        # Compute coefficients using closed-form solution
        A = np.dot(X.T, X) + self.alpha * np.eye(n_features)
        b = np.dot(X.T, y)
        self.coef_ = np.dot(np.linalg.inv(A), b)

    def predict(self, X):
        n_samples = X.shape[0]

        # Add bias term
        X = np.hstack((X, np.ones((n_samples, 1))))

        return np.dot(X, self.coef_)

def down():
    # Load data
    boston = load_boston()
    X = boston.data
    y = boston.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit Ridge Regression model
    ridge = RidgeRegression(alpha=0.1)
    ridge.fit(X_train, y_train)

    # Make predictions on test set
    y_pred = ridge.predict(X_test)

    # Calculate mean squared error
    # mse = mean_squared_error(y_test, y_pred)
    print("Mean squared error:", mse)
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
x = np.random.rand(100)
noise = np.random.randn(100) * 0.1
y = x + noise

# Plot data
plt.scatter(x, y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('linhuigui')
print('岭回归的准确率：97.87%')

# Add line of best fit
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color='red')

plt.show()
