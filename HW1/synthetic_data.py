# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# %%
# generate
x1 = np.random.uniform(low=50, high=80, size=100)
x2 = np.random.uniform(low=10, high=40, size=100)
y = x1 + x2 + np.random.normal(size=100)

dataX = np.vstack((x1, x2)).transpose()
dataY = y.copy()
dataY = dataY.reshape(dataY.shape[0], 1)

# %%
# show
plt.plot(x1, y, 'o', color='blue')
plt.plot(x2, y, 'o', color='red')
plt.show()

# %%
# normalize
scaler_x = StandardScaler()
scaler_y = StandardScaler()

dataX = scaler_x.fit_transform(dataX)
dataY = scaler_y.fit_transform(dataY)

# %%
# train test split
trainX, testX, trainY, testY = train_test_split(dataX, dataY, train_size=0.7, random_state=314, shuffle=True)


# %%
# train function
def linear_train(the_x, the_y, learning_rate=0.1, iterations=1000, epsilon=10e-6, verbose=True):
    # initialize
    the_x = the_x.copy()
    the_y = the_y.copy()
    num_features = the_x.shape[1]
    num_obs = the_x.shape[0]

    # prepare for train
    the_x = np.hstack((np.ones((num_obs, 1)), the_x))
    w = np.zeros((num_features + 1, 1))
    adagrad = np.zeros((num_features + 1, 1))

    # train
    for i in range(iterations):
        grad = -2 * np.dot(the_x.transpose(), the_y - np.dot(the_x, w))
        adagrad += grad ** 2
        w = w - learning_rate * grad / np.sqrt(adagrad + epsilon)

        if verbose:
            if i % 100 == 0:
                cur_mse = np.sqrt(np.sum(np.power(the_y - the_x.dot(w), 2)))
                print(i, " :"'MSE:', cur_mse)

    return w


# %%
# train
w1 = linear_train(the_x=trainX,
                  the_y=trainY,
                  learning_rate=0.001,
                  iterations=1000000)

#%%
# sklearn linear regression
skLearn_linear = LinearRegression()
skLearn_linear.fit(trainX, trainY)
