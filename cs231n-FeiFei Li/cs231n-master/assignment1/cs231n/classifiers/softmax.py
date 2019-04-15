import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #print(num_train)
  for i in range(num_train):
    score = np.dot(X[i],W)
    score = score - np.max(score)
    score_exp = np.exp(score)
    correct_class_score = score_exp[y[i]]
    Li = -np.log(correct_class_score/np.sum(score_exp))
    loss = loss + Li 
    dW[:, y[i]] -= X[i]
    for j in range(num_classes):
      dW[:,j] += (score_exp[j] / np.sum(score_exp)) * X[i] 
  loss /= num_train
  dW /= num_train
  loss = loss + 0.5 * reg * np.sum(W*W)
  dW = dW + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #print(num_train)  
  
  score = np.dot(X,W)   # (N,C)
  score = score - np.max(score,axis = 1)[:,np.newaxis]
  score_exp = np.exp(score)  
  correct_class_score = score_exp[np.arange(num_train),y] #(N,1)
  Li = -np.log(correct_class_score/np.sum(score_exp,axis = 1))    #(N,1) 
  loss = np.sum(Li) 
  loss /= num_train
  loss_selector = np.zeros_like(score_exp) #N*C
  loss_selector[np.arange(num_train),y] = 1.0
  dW = np.dot(X.T, (score_exp/ np.sum(score_exp,axis = 1,keepdims= True))-loss_selector)
  dW /= num_train
  loss = loss +  0.5 *reg * np.sum(W*W)
  dW = dW + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

