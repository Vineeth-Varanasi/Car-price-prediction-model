import numpy as np
import math

class StatisticalMethods:
  ## checks for errors
    def MSE(self , Y , Y_predicted):

        return np.mean((Y -Y_predicted)**2)
        # returns the MSE of the model

    def RMSE(self , Y , Y_predicted):
        return  (np.mean((Y -Y_predicted)**2))**0.5

# performance metric
    def Rsquared(self , Y , Y_predicted):
        return 1 - np.sum( (Y - Y_predicted)**2 ) /np.sum((Y_predicted - np.mean(Y))**2)