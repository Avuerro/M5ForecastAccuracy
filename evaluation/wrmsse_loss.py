import numpy as np
import tensorflow as tf
import keras.backend as K


class WRMSSE_Loss:
  """
    Calculates WRMSE based on implementation of 
    https://www.kaggle.com/sibmike/fast-clear-wrmsse-18ms
    Params:
      SW = sales weights..eve
      roll_mat_csr = sparse tensor ..
  """
  def __init__(self, SW, roll_mat_csr):
    self.SW = SW
    self.roll_mat_csw = self.convert_sparse_matrix_to_sparse_tensor(roll_mat_csr)

  ## Based on https://stackoverflow.com/questions/40896157/scipy-sparse-csr-matrix-to-tensorflow-sparsetensor-mini-batch-gradient-descent
  def convert_sparse_matrix_to_sparse_tensor(self,X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)
  
  def rollup_loss(self,v):
    intermediate = tf.sparse.sparse_dense_matmul(self.roll_mat_csw,tf.transpose(v))
    return intermediate
    
  # Function to calculate WRMSSE during training:
  # make sure that s = S, w = W, sw=SW) are available
  def wrmsse_loss(self,y_true, y_pred):
      '''
      preds - Predictions: pd.DataFrame of size (30490 rows, N day columns)
      y_true - True values: pd.DataFrame of size (30490 rows, N day columns)
      sequence_length - np.array of size (42840,)
      sales_weight - sales weights based on last 28 days: np.array (42840,)
      '''
          
      return K.sum(
              K.sqrt(
                  K.mean(
                      K.square(self.rollup_loss(y_pred-y_true))
                          ,axis=1)) * self.SW)/12 #<-used to be mistake here