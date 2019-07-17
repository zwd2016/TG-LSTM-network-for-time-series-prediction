# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 10:31:40 2018

@author: Wendong Zheng
"""

from math import sqrt
from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics
from keras import regularizers
from keras import optimizers
from ind_rnn import IndRNN
from keras.layers.normalization import BatchNormalization
from custom_layers import LSTM_Custom

np.random.seed(1337)  # for reproducibility
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
# load dataset
dataset = read_csv('pollution_pm2.5.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())
# split into train and test sets
values = reframed.values
n_train_hours = 548 * 24#365*24*2=2years,548*24=1.5years
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
test_X1, test_y1 = test[:, :-1], test[:, -1]
test_X2, test_y2 = test[:, :-1], test[:, -1]
test_X3, test_y3 = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
test_X1 = test_X1.reshape((test_X1.shape[0], 1, test_X1.shape[1]))
test_X2 = test_X2.reshape((test_X2.shape[0], 1, test_X2.shape[1]))
test_X3 = test_X3.reshape((test_X3.shape[0], 1, test_X3.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)



#1 layer
# design network LSTM
print('Build LSTM model...')
model = Sequential()
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mae', optimizer='adam',metrics=['mae'])
# fit network
history = model.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# design network TG-LSTM
print('Build Our model...')
model1 = Sequential()
model1.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]),implementation=2))
model1.add(Dense(1))
model1.compile(loss='mae', optimizer='adam',metrics=['mae'])
# fit network
history1 = model1.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X1, test_y1), verbose=2, shuffle=False)

#IndRNN
print('Build IndRNN model...')
model2 = Sequential()
model2.add(IndRNN(128, input_shape=(train_X.shape[1], train_X.shape[2]),recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0
                 ))
model2.add(Dense(1, activation='sigmoid'))
# try using different optimizers and different optimizer configs
model2.compile(loss='mae',optimizer='adam',metrics=['mae'])            
history2 = model2.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X2, test_y2), verbose=2, shuffle=False)

# design network LSTM+zoneout
print('Build LSTM+Zoneout model...')
model3 = Sequential()
model3.add(LSTM_Custom(128, zoneout_c=0.5, zoneout_h=0.05,dropout=0.2,
    input_shape=(train_X.shape[1], train_X.shape[2])))#unit_size=128
model3.add(Dense(1))
model3.compile(loss='mae',optimizer='adam',metrics=['mae'])
# fit network
history3 = model3.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X3, test_y3), verbose=2, shuffle=False)



'''
#2-layer
# design network LSTM
print('Build LSTM model...')
model = Sequential()
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mae', optimizer='adam',metrics=['mae'])
# fit network
history = model.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# design network LSTM-modify
print('Build Our model...')
model1 = Sequential()
model1.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]),recurrent_dropout=0.1,implementation=2,return_sequences=True))
model1.add(LSTM(128,implementation=2,return_sequences=False,recurrent_dropout=0.1))
model1.add(Dense(1))
model1.compile(loss='mae', optimizer='adam',metrics=['mae'])
# fit network
history1 = model1.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X1, test_y1), verbose=2, shuffle=False)

#IndRNN
print('Build IndRNN model...')
model2 = Sequential()
model2.add(IndRNN(128, input_shape=(train_X.shape[1], train_X.shape[2]),recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=False))#默认值dropout=0.0, recurrent_dropout=0.0,用先前研究提到的6层IndRNN
model2.add(Dense(1, activation='sigmoid'))
# try using different optimizers and different optimizer configs
model2.compile(loss='mae',optimizer='adam',metrics=['mae'])            
history2 = model2.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X2, test_y2), verbose=2, shuffle=False)

# design network LSTM+zoneout
print('Build LSTM+Zoneout model...')
model3 = Sequential()
model3.add(LSTM_Custom(128, zoneout_c=0.5, zoneout_h=0.05,dropout=0.3,return_sequences=True,
    input_shape=(train_X.shape[1], train_X.shape[2])))#unit_size=128
model3.add(LSTM_Custom(128, zoneout_c=0.5, zoneout_h=0.05,dropout=0.3,return_sequences=False))
model3.add(Dense(1))
model3.compile(loss='mae',optimizer='adam',metrics=['mae'])
# fit network
history3 = model3.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X3, test_y3), verbose=2, shuffle=False)
'''


'''
#6-layer
# design network LSTM
print('Build LSTM model...')
model = Sequential()
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mae', optimizer='adam',metrics=['mae'])
# fit network
history = model.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# design network LSTM-modify
print('Build Our model...')
model1 = Sequential()
model1.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]),implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128,implementation=2,return_sequences=False))
model1.add(Dense(1))
model1.compile(loss='mae', optimizer='adam',metrics=['mae'])
# fit network
history1 = model1.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X1, test_y1), verbose=2, shuffle=False)

#IndRNN
print('Build IndRNN model...')
model2 = Sequential()
model2.add(IndRNN(128, input_shape=(train_X.shape[1], train_X.shape[2]),recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=False))#默认值dropout=0.0, recurrent_dropout=0.0,用先前研究提到的6层IndRNN
model2.add(Dense(1, activation='sigmoid'))
# try using different optimizers and different optimizer configs
model2.compile(loss='mae',optimizer='adam',metrics=['mae'])            
history2 = model2.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X2, test_y2), verbose=2, shuffle=False)

# design network LSTM+zoneout
print('Build LSTM+Zoneout model...')
model3 = Sequential()
model3.add(LSTM_Custom(128, zoneout_c=0.5, zoneout_h=0.05,dropout=0.3,return_sequences=True,
    input_shape=(train_X.shape[1], train_X.shape[2])))#unit_size=128
model3.add(LSTM_Custom(128, zoneout_c=0.5, zoneout_h=0.05,dropout=0.3,return_sequences=True,
    input_shape=(train_X.shape[1], train_X.shape[2])))#unit_size=128
model3.add(LSTM_Custom(128, zoneout_c=0.5, zoneout_h=0.05,dropout=0.3,return_sequences=True,
    input_shape=(train_X.shape[1], train_X.shape[2])))#unit_size=128
model3.add(LSTM_Custom(128, zoneout_c=0.5, zoneout_h=0.05,dropout=0.3,return_sequences=True,
    input_shape=(train_X.shape[1], train_X.shape[2])))#unit_size=128
model3.add(LSTM_Custom(128, zoneout_c=0.5, zoneout_h=0.05,dropout=0.3,return_sequences=True,
    input_shape=(train_X.shape[1], train_X.shape[2])))#unit_size=128
model3.add(LSTM_Custom(128, zoneout_c=0.5, zoneout_h=0.05,dropout=0.3,return_sequences=False))
model3.add(Dense(1))
model3.compile(loss='mae',optimizer='adam',metrics=['mae'])
# fit network
history3 = model3.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X3, test_y3), verbose=2, shuffle=False)
'''

'''
#21-layer
# design network LSTM
print('Build LSTM model...')
model = Sequential()
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=False))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mae', optimizer='adam',metrics=['mae'])
# fit network
history = model.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# design network LSTM-modify
print('Build Our model...')
model1 = Sequential()
model1.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]),implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128, implementation=2,return_sequences=True))
model1.add(LSTM(128,implementation=2,return_sequences=False))
model1.add(Dense(1))
model1.compile(loss='mae', optimizer='adam',metrics=['mae'])
# fit network
history1 = model1.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X1, test_y1), verbose=2, shuffle=False)

#IndRNN
print('Build IndRNN model...')
model2 = Sequential()
model2.add(IndRNN(128, input_shape=(train_X.shape[1], train_X.shape[2]),recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.2, recurrent_dropout=0.1,
                 return_sequences=True))
model2.add(IndRNN(128, recurrent_clip_min=-1, recurrent_clip_max=-1, dropout=0.0, recurrent_dropout=0.0,
                 return_sequences=False))#默认值dropout=0.0, recurrent_dropout=0.0,用先前研究提到的6层IndRNN
model2.add(Dense(1, activation='sigmoid'))
# try using different optimizers and different optimizer configs
model2.compile(loss='mae',optimizer='adam',metrics=['mae'])            
history2 = model2.fit(train_X, train_y, epochs=20, batch_size=130, validation_data=(test_X2, test_y2), verbose=2, shuffle=False)
'''

# plot history train-loss
pyplot.ylabel("Train loss value")  
pyplot.xlabel("The number of epochs")  
pyplot.title("Loss function-epoch curves")
pyplot.plot(history.history['loss'], label='train_LSTM')
pyplot.plot(history2.history['loss'], label='train_IndRNN')
pyplot.plot(history3.history['loss'], label='train_LSTM+Zoneout')
pyplot.plot(history1.history['loss'], label='train_Our')
pyplot.legend()
pyplot.savefig('Figure-PM 2.5-train-loss.png', dpi=300)
pyplot.show()

# plot history val-loss
pyplot.ylabel("Validation Loss value")  
pyplot.xlabel("The number of epochs")  
pyplot.title("Loss function-epoch curves")
pyplot.plot(history.history['val_loss'], label='val_LSTM')
pyplot.plot(history2.history['val_loss'], label='val_IndRNN')
pyplot.plot(history3.history['val_loss'], label='val_LSTM+Zoneout')
pyplot.plot(history1.history['val_loss'], label='val_Our')
pyplot.legend()
pyplot.savefig('Figure-PM 2.5-val-loss.png', dpi=300)
pyplot.show()

# make a prediction LSTM
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# make a prediction LSTM-Our
yhat1 = model1.predict(test_X1)
test_X1 = test_X1.reshape((test_X1.shape[0], test_X1.shape[2]))

# make a prediction IndRNN
yhat2 = model2.predict(test_X2)
test_X2 = test_X2.reshape((test_X2.shape[0], test_X2.shape[2]))

# make a prediction Zoneout
yhat3 = model3.predict(test_X3)
test_X3 = test_X3.reshape((test_X3.shape[0], test_X3.shape[2]))

# invert scaling for forecast LSTM
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for forecast LSTM-Our
inv_yhat1 = concatenate((yhat1, test_X1[:, 1:]), axis=1)
inv_yhat1 = scaler.inverse_transform(inv_yhat1)
inv_yhat1 = inv_yhat1[:,0]

# invert scaling for forecast IndRNN
inv_yhat2 = concatenate((yhat2, test_X2[:, 1:]), axis=1)
inv_yhat2 = scaler.inverse_transform(inv_yhat2)
inv_yhat2 = inv_yhat2[:,0]

# invert scaling for forecast Zoneout
inv_yhat3 = concatenate((yhat3, test_X3[:, 1:]), axis=1)
inv_yhat3 = scaler.inverse_transform(inv_yhat3)
inv_yhat3 = inv_yhat3[:,0]

# invert scaling for actual LSTM
inv_y = scaler.inverse_transform(test_X)
inv_y = inv_y[:,0]

# invert scaling for actual LSTM-Our
inv_y1 = scaler.inverse_transform(test_X1)
inv_y1 = inv_y1[:,0]

# invert scaling for actual IndRNN
inv_y2 = scaler.inverse_transform(test_X2)
inv_y2 = inv_y2[:,0]

# invert scaling for actual Zoneout
inv_y3 = scaler.inverse_transform(test_X3)
inv_y3 = inv_y3[:,0]

# calculate RMSE and MAE LSTM
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
mae = mean_absolute_error(inv_y, inv_yhat)
print('LSTM Test RMSE: %.3f' % rmse)
print('LSTM Test MAE: %.3f' % mae)

# calculate RMSE and MAE IndRNN
rmse2 = sqrt(mean_squared_error(inv_y2, inv_yhat2))
mae2 = mean_absolute_error(inv_y2, inv_yhat2)
print('IndRNN Test RMSE: %.3f' % rmse2)
print('IndRNN Test MAE: %.3f' % mae2)

# calculate RMSE and MAE Zoneout
rmse3 = sqrt(mean_squared_error(inv_y3, inv_yhat3))
mae3 = mean_absolute_error(inv_y3, inv_yhat3)
print('LSTM+Zoneout Test RMSE: %.3f' % rmse3)
print('LSTM+Zoneout Test MAE: %.3f' % mae3)

# calculate RMSE and MAE Our
rmse1 = sqrt(mean_squared_error(inv_y1, inv_yhat1))
mae1 = mean_absolute_error(inv_y1, inv_yhat1)
print('Our method Test RMSE: %.3f' % rmse1)
print('Our method Test MAE: %.3f' % mae1)

pyplot.figure(figsize=(20,10))
pyplot.title('PM 2.5(the next 96 hours)')
pyplot.xlabel('Time range(h)')
pyplot.ylabel(' PM2.5 range')
pyplot.plot(inv_y[:96],label='true')
pyplot.plot(inv_yhat[:96],'r--',label='predictions_LSTM')
pyplot.plot(inv_yhat2[:96],'c-.',label='predictions_IndRNN')
pyplot.plot(inv_yhat3[:96],'k:',label='predictions_LSTM+Zoneout')
pyplot.plot(inv_yhat1[:96],'g-*',label='predictions_Our')
pyplot.legend()
pyplot.savefig('Figure-PM 2.5.png', dpi=300)
pyplot.show()