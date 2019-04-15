import math
import time
import numpy as np
X = np.ndfromtxt('logistic_x.txt')
y = np.ndfromtxt("logistic_y.txt")
#print (X.shape)
#print (y.shape)

#print (theta.shape)
def h_vec(theta, X_train):
    return 1 / (1 + np.exp(np.dot(X_train, theta)))

  
def GD_vec(theta, X_train, y_train, alpha):
    theta -= alpha * np.matmul(np.transpose(X_train), (h_vec(theta, X_train) - y_train))
      
      
def train_vec(X_train, y_train, max_iter, alpha, theta):
    #theta = np.zeros([X_train.shape[1]])
    for i in range(max_iter):
        GD_vec(theta, X_train, y_train, alpha)  
    return theta 
    
def g_vec(x):
    return 1 / (1 + np.exp(x))

def g_pi_vec(x):
    return np.matmul(g_vec(x),(1-g_vec(x)))
    
def NT_vec(X_train, y_train, theta):
        z = np.dot(np.matmul(X_train , theta), y_train)
        Hess_l = np.dot(np.matmul((1/X_train.shape[0]) * X_train , (g_vec(z)-1)), y_train)
        D = (1/X_train.shape[0])* g_pi_vec(z)
        H = np.transpose(X_train) * D * X_train
        theta -= np.dot(H.T, Hess_l)
            
    
def train_NT_vec(X_train, y_train, max_iter, alpha, theta):
    for i in range(max_iter):
        NT_vec(X_train, y_train, theta)
        print (theta)
    return theta
    

theta = np.zeros([X.shape[1]])    
max_iter = 100
alpha = 0.001
start = time.time()
theta = train_vec(X, y, max_iter, alpha, theta)
end = time.time()   
print (theta)

theta = np.zeros([X.shape[1]])
start = time.time()
theta = train_NT_vec(X, y, max_iter, alpha, theta)
end = time.time() 
#print (h_vec(theta, X)- y)
print (theta)