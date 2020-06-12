
import keras.backend as K



## not sure about correctness of this...

def root_mean_squared_scaled_error(y_true,y_pred):
    sample_length = 28
    forecasting_horizon = 1
    upper_bound = sample_length + forecasting_horizon
    lower_bound = sample_length + 1 - 1
    numerator =  K.sum(K.square(y_true[lower_bound:upper_bound]-y_pred[lower_bound:upper_bound]))
    lower_bound = 2 - 1
    denominator = (1/(sample_length - 1 )) * K.sum(K.square(y_true[lower_bound:sample_length] - y_true[lower_bound-1:sample_length-1]))
    value_to_be_sqrt = (1/forecasting_horizon) * (numerator/denominator)
    result = K.sqrt(value_to_be_sqrt)
    return result



