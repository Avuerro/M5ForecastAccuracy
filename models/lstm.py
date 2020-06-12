from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional



def lstm_model(timesteps,n_features,optimizer='adam',loss='mse',metrics=['acc']):
  model = Sequential()
  model.add(Bidirectional(LSTM(20, return_sequences=True, input_shape=(timesteps, n_features))))
  model.add(Bidirectional(LSTM(10)))
  model.add(Dense(30490))
  model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
  return model