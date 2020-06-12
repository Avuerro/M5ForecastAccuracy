import numpy as np
import pandas as pd



def create_windows(raw_data,timesteps,prediction_steps):
  
  len_window = timesteps + prediction_steps 
  nr_training_days = raw_data.shape[0]
  nr_sets = nr_training_days - len_window + 1

  base, predictions = [], []

  for i in range(nr_sets):
      samples = raw_data.iloc[i:i+timesteps]
      pred = raw_data.iloc[i+timesteps]
      base.append(samples.to_numpy())
      predictions.append(pred.to_numpy())
      
  X = np.array(base)
  y = np.array(predictions)

  del base, predictions



  return X,y