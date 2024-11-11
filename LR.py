import numpy as np
class LinearRegression:
    

    def __init__ (self , lr = 0.005 , n_iterations  = 500 , regularization_constant  = 30):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.costs = [] ## keeps track of a MSE in each iteration (which will be used later in the cost function vs epoch graph)
        self.regularization_constant = regularization_constant

    def fit(self, X,y):
        number_of_samples , number_of_features  = X.shape
        #starting off the gradient descent at w1,w2,...,w34 and bias at 0
        self.weights = np.zeros(number_of_features)
        self.bias = 0 
        #gradient descent for 1000 iterations
        
        for _ in range(self.n_iterations):

            y_predict  = np.dot(X , self.weights) + self.bias # we took X and self weights in that specific order since it is a matrix multiplication

            dw  = (1/number_of_samples)*(  np.dot(X.T , y_predict  - y)) + (self.regularization_constant/number_of_samples)*(self.weights)
            # here we had to take a transpose because the shape of X is 183,34 and y predict is 183, and for matrix multiplication
            # we need 34,183 * 183, so we took transpose of matrix X
            #self.regularization constant is the value of lambda
            
            db  = (1/number_of_samples)*np.sum(y_predict  - y )
            #updating the weights
            
            self.weights =self.weights - self.lr*dw
            self.bias= self.bias - self.lr*db
            cost = (1 / (2 * number_of_samples)) * np.sum((y_predict - y) ** 2) + \
                   (self.regularization_constant / (2 * number_of_samples)) * np.sum(self.weights ** 2)
            self.costs.append(cost) 


    def predict(self,X):
        predictions  = np.dot(X,self.weights) +self.bias
        return predictions

