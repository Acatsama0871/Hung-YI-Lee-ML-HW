# %%
import os
import numpy as np
import pandas as pd
from math import floor
from sklearn.preprocessing import StandardScaler

# %%
# load data
# path
cwd = os.getcwd()
#import the data
dataTrain = pd.read_csv(os.path.join(cwd, 'data/train.csv'),
                        encoding='big5')
dataTest = pd.read_csv(os.path.join(cwd, 'data/test.csv'),
                       encoding='big5', header=None)

# dataTrain = pd.read_csv(os.path.join(cwd, 'HW1/data/train.csv'),
#                         encoding='big5')
# dataTest = pd.read_csv(os.path.join(cwd, 'HW1/data/test.csv'),
#                        encoding='big5', header=None)


# %%
def transpose_data(df, is_train=True, NRtoZero=True):
    # initialize
    the_df = df.copy()
    the_delta = len(the_df.iloc[:, 0].unique())

    # remove
    the_df = the_df.drop(the_df.columns[0], axis=1)
    if is_train:
        col_names = list(the_df.iloc[:, 1].unique())
        the_df = the_df.drop(the_df.columns[[0, 1]], axis=1)
    else:
        col_names = list(the_df.iloc[:, 0].unique())
        the_df = the_df.drop(the_df.columns[0], axis=1)

    # append
    result_df = the_df.iloc[range(18), :].transpose().values
    for i in range(1, the_delta):
        cur_df = the_df.iloc[range(18 * i, 18 * (i + 1)), :].transpose().values
        result_df = np.vstack((result_df, cur_df))

    # add column name
    result_df = pd.DataFrame(result_df)
    result_df.columns = col_names

    if NRtoZero:
        result_df['RAINFALL'][result_df['RAINFALL'] == 'NR'] = 0

    result_df = result_df.astype(np.float)

    return result_df.pop('PM2.5'), result_df


#%%
Y, X = transpose_data(dataTrain)
testY, testX = transpose_data(dataTest, is_train=False)

#%%
# split to train, validate
train_size = floor(0.8 * X.shape[0])
trainY = Y[:train_size].values
trainX = X.iloc[:train_size, :].values
validY = Y[train_size:].values
validX = X[train_size:].values

scaler1 = StandardScaler()
scaler2 = StandardScaler()
trainX = scaler1.fit_transform(trainX)
trainY = trainY.reshape(trainY.shape[0], 1)
trainY = scaler2.fit_transform(trainY)

#%%
# prepare
w = np.zeros((18, 1))
trainX_model = np.hstack((np.ones([trainX.shape[0], 1]), trainX))
trainY_model = trainY.reshape(trainY.shape[0], 1)
adagrad = np.zeros((18, 1))
learning_rate = 0.001
iterations = 100000
eps = 10e-6

#%%
for i in range(iterations):
    grad = -2 * np.dot(trainX_model.transpose(), trainY_model - np.dot(trainX_model, w))
    adagrad += grad**2
    w = w - learning_rate * grad / np.sqrt(adagrad + eps)
    if i % 100 == 0:
        cur_mse = np.sqrt(np.sum(np.power(trainY_model - trainX_model.dot(w), 2)))
        print('MSE:', cur_mse)

#%%
validX_pred = np.hstack((np.ones([validX.shape[0], 1]), validX))
validY_pred = validY.reshape(validY.shape[0], 1)
model_pred = validX_pred.dot(w)
