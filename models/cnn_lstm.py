from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Conv1D, MaxPooling1D, Flatten, TimeDistributed, Dropout


# def cnn_lstm_model(timesteps, n_features, n_outputs, optimizer='adam',loss='mse',metrics=['mse']):

#   model = Sequential()
#   model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(timesteps, n_features)))
#   model.add(MaxPooling1D(pool_size=2))
#   model.add(Bidirectional(LSTM(20, activation='relu', return_sequences=True)))
#   model.add(Bidirectional(LSTM(10, activation='relu')))
#   model.add(Dense(n_outputs))
#   model.compile(optimizer=optimizer, loss= loss)

#   return model


## paper + dropout variant
def cnn_lstm_model(timesteps, n_features, n_outputs, optimizer='adam',loss='mse',metrics=['mse']):

  model = Sequential()
  model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(timesteps, n_features)))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
  # model.add(Dropout(0.3))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Bidirectional(LSTM(20, activation='relu', return_sequences=True)))
  model.add(Bidirectional(LSTM(10, activation='relu')))
  model.add(Dense(n_outputs))
  model.compile(optimizer=optimizer, loss= loss)

  return model