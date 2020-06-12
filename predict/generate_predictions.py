import numpy as np
import pandas as pd

def predict_sales(sales,timesteps,scaler, model, nr_of_days):

  # get input for prediction by selecting last 28 days from sales
  X_pred = []
  X_pred.append(sales.iloc[-timesteps:].to_numpy())
  X_pred = np.array(X_pred)
  
  # get prediction
  prediction = model.predict(X_pred)
  
  # add prediction to sales so that it can be used for next prediction
  sales.loc[sales.shape[0]] = prediction[0]
    
  predictions = sales.iloc[-28:]
  predictions = scaler.inverse_transform(predictions)
  predictions = np.round(np.abs(predictions))
  predictions = pd.DataFrame(predictions).T
  return predictions