import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

# Generate random data
np.random.seed(42)
n_samples, n_features = 50, 10
X = np.random.randn(n_samples, n_features)
w = np.zeros(n_features)
relevant_features = np.random.randint(0, n_features, 5)
for i in relevant_features:
    w[i] = np.random.randn()

y = np.dot(X, w) + np.random.randn(n_samples) * 0.5

# Fit Lasso Regression model
alpha = 0.1
lasso = Lasso(alpha=alpha)
lasso.fit(X, y)

# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.stem(w)
plt.title('True coefficients')
plt.subplot(122)
plt.stem(lasso.coef_)
plt.title('Lasso coefficients (alpha = %s)' % alpha)
plt.show()
'''生成了一个随机数据集，并使用Lasso回归拟合数据。我们使用Matplotlib库绘制了两个图形，一个显示真实系数，另一个显
示Lasso回归的系数。Lasso回归的系数显示了哪些特征被选择，哪些特征被忽略。在这个例子中，我们使用了10个特征，
但只有5个特征被选择。'''