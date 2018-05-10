# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 19:10:25 2018

@author: Ganesh
"""
######################################################
# DL functions
######################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import sklearn
import sklearn.datasets

# Conmpute the sigmoid of a vector
def sigmoid(Z):  
    A=1/(1+np.exp(-Z))    
    cache=Z
    return A,cache

# Conmpute the Relu of a vector
def relu(Z):
    A = np.maximum(0,Z)
    cache=Z
    return A,cache

# Conmpute the tanh of a vector
def tanh(Z):
    A = np.tanh(Z)
    cache=Z
    return A,cache

# Conmpute the softmax of a vector
def softmax(Z):  
    # get unnormalized probabilities
    exp_scores = np.exp(Z.T)
    # normalize them for each example
    A = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)   
    cache=Z
    return A,cache

# Conmpute the softmax of a vector
def stableSoftmax(Z):  
    #Compute the softmax of vector x in a numerically stable way.
    shiftZ = Z.T - np.max(Z.T,axis=1).reshape(-1,1)
    exp_scores = np.exp(shiftZ)

    # normalize them for each example
    A = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
    cache=Z
    return A,cache

# Compute the detivative of Relu 
def reluDerivative(dA, cache):
  
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.  
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0 
    return dZ

# Compute the derivative of sigmoid
def sigmoidDerivative(dA, cache):
    Z = cache  
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)   
    return dZ

# Compute the derivative of tanh
def tanhDerivative(dA, cache):
    Z = cache  
    a = np.tanh(Z)
    dZ = dA * (1 - np.power(a, 2))  
    return dZ

# Compute the derivative of softmax
def softmaxDerivative(dA, cache,y,numTraining):
      # Note : dA not used. dL/dZ = dL/dA * dA/dZ = pi-yi
      Z = cache 
      # Compute softmax
      exp_scores = np.exp(Z.T)
      # normalize them for each example
      probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  

      # compute the gradient on scores
      dZ = probs

      # dZ = pi- yi
      dZ[range(int(numTraining)),y[:,0]] -= 1
      return(dZ)

# Compute the derivative of softmax
def stableSoftmaxDerivative(dA, cache,y,numTraining):
      # Note : dA not used. dL/dZ = dL/dA * dA/dZ = pi-yi
      Z = cache 
      # Compute stable softmax
      shiftZ = Z.T - np.max(Z.T,axis=1).reshape(-1,1)
      exp_scores = np.exp(shiftZ)
      # normalize them for each example
      probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  
      #print(probs)      
      # compute the gradient on scores
      dZ = probs

      # dZ = pi- yi
      dZ[range(int(numTraining)),y[:,0]] -= 1
      return(dZ)
      
      
# Initialize the model 
# Input : number of features
#         number of hidden units
#         number of units in output
# Returns: Weight and bias matrices and vectors
def initializeModel(numFeats,numHidden,numOutput):
    np.random.seed(1)
    W1=np.random.randn(numHidden,numFeats)*0.01 #  Multiply by .01 
    b1=np.zeros((numHidden,1))
    W2=np.random.randn(numOutput,numHidden)*0.01
    b2=np.zeros((numOutput,1))
    
    # Create a dictionary of the neural network parameters
    nnParameters={'W1':W1,'b1':b1,'W2':W2,'b2':b2}
    return(nnParameters)


# Initialize model for L layers
# Input : List of units in each layer
# Returns: Initial weights and biases matrices for all layers
def initializeDeepModel(layerDimensions):
    np.random.seed(3)
    # note the Weight matrix at layer 'l' is a matrix of size (l,l-1)
    # The Bias is a vectors of size (l,1)
    
    # Loop through the layer dimension from 1.. L
    layerParams = {}
    for l in range(1,len(layerDimensions)):
         layerParams['W' + str(l)] = np.random.randn(layerDimensions[l],layerDimensions[l-1])*0.01 #  Multiply by .01 
         layerParams['b' + str(l)] = np.zeros((layerDimensions[l],1))        
   
    return(layerParams)
    return Z, cache

# He Initialization model for L layers
# Input : List of units in each layer
# Returns: Initial weights and biases matrices for all layers
# He initilization multiplies the random numbers with sqrt(2/layerDimensions[l-1])
def HeInitializeDeepModel(layerDimensions):
    np.random.seed(3)
    # note the Weight matrix at layer 'l' is a matrix of size (l,l-1)
    # The Bias is a vectors of size (l,1)
    
    # Loop through the layer dimension from 1.. L
    layerParams = {}
    for l in range(1,len(layerDimensions)):
         layerParams['W' + str(l)] = np.random.randn(layerDimensions[l],
                       layerDimensions[l-1])*np.sqrt(2/layerDimensions[l-1]) 
         layerParams['b' + str(l)] = np.zeros((layerDimensions[l],1))        
   
    return(layerParams)
    return Z, cache

# Xavier Initialization model for L layers
# Input : List of units in each layer
# Returns: Initial weights and biases matrices for all layers
# Xavier initilization multiplies the random numbers with sqrt(1/layerDimensions[l-1])
def XavInitializeDeepModel(layerDimensions):
    np.random.seed(3)
    # note the Weight matrix at layer 'l' is a matrix of size (l,l-1)
    # The Bias is a vectors of size (l,1)
    
    # Loop through the layer dimension from 1.. L
    layerParams = {}
    for l in range(1,len(layerDimensions)):
         layerParams['W' + str(l)] = np.random.randn(layerDimensions[l],
                       layerDimensions[l-1])*np.sqrt(1/layerDimensions[l-1]) 
         layerParams['b' + str(l)] = np.zeros((layerDimensions[l],1))        
   
    return(layerParams)
    return Z, cache

# Compute the activation at a layer 'l' for forward prop in a Deep Network
# Input : A_prec - Activation of previous layer
#         W,b - Weight and bias matrices and vectors
#         activationFunc - Activation function - sigmoid, tanh, relu etc
# Returns : The Activation of this layer
#         : 
# Z = W * X + b
# A = sigmoid(Z), A= Relu(Z), A= tanh(Z)
def layerActivationForward(A_prev, W, b, keep_prob=1, activationFunc="relu"):
    
    # Compute Z
    Z = np.dot(W,A_prev) + b
    forward_cache = (A_prev, W, b) 
    # Compute the activation for sigmoid
    if activationFunc == "sigmoid":
        A, activation_cache = sigmoid(Z)  
    # Compute the activation for Relu
    elif activationFunc == "relu":  
        A, activation_cache = relu(Z)
    # Compute the activation for tanh
    elif activationFunc == 'tanh':
        A, activation_cache = tanh(Z)  
    elif activationFunc == 'softmax':
        A, activation_cache = stableSoftmax(Z)  
    
    cache = (forward_cache, activation_cache)
    return A, cache

# Compute the forward propagation for layers 1..L
# Input : X - Input Features
#         paramaters: Weights and biases
#         hiddenActivationFunc - Activation function at hidden layers Relu/tanh
#         outputActivationFunc - Activation function at output - sigmoid/softmax
# Returns : AL 
#           caches
# The forward propoagtion uses the Relu/tanh activation from layer 1..L-1 and sigmoid actiovation at layer L
def forwardPropagationDeep(X, parameters,keep_prob=1, hiddenActivationFunc='relu',outputActivationFunc='sigmoid'):
    caches = []
    #initialize the dropout matrix
    dropoutMat = {}
    # Set A to X (A0)
    A = X
    L = int(len(parameters)/2) # number of layers in the neural network
    # Loop through from layer 1 to upto layer L
    for l in range(1, L):
        A_prev = A 
        # Zi = Wi x Ai-1 + bi  and Ai = g(Zi)
        A, cache = layerActivationForward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], keep_prob, activationFunc = hiddenActivationFunc)
        
        # Randomly drop some activation units
        # Create a matrix as the same shape as A
        D = np.random.rand(A.shape[0],A.shape[1])     
        D = (D < keep_prob) 
        # We need to use the same 'dropout' matrix in backward propagation
        # Save the dropout matrix for use in backprop
        dropoutMat["D" + str(l)] =D
        A= np.multiply(A,D)                                      
        A = np.divide(A,keep_prob) 
      
        caches.append(cache)

    
    # Since this is binary classification use the sigmoid activation function in
    # last layer   
    AL, cache = layerActivationForward(A, parameters['W'+str(L)], parameters['b'+str(L)], activationFunc = outputActivationFunc)
    caches.append(cache)
            
    return AL, caches, dropoutMat


# Compute the cost
# Input : Activation of last layer
#       : Output from data
#       : Y
#       :outputActivationFunc - Activation function at output - sigmoid/softmax
# Output: cost
def computeCost(parameters,AL,Y,outputActivationFunc="sigmoid"):
    if outputActivationFunc=="sigmoid":
        m= float(Y.shape[1])
        # Element wise multiply for logprobs
        cost=-1/m *np.sum(Y*np.log(AL) + (1-Y)*(np.log(1-AL)))
        cost = np.squeeze(cost) 
    elif outputActivationFunc=="softmax":
        # Take transpose of Y for softmax
        Y=Y.T
        m= float(len(Y))
        # Compute log probs. Take the log prob of correct class based on output y
        correct_logprobs = -np.log(AL[range(int(m)),Y.T])
        # Conpute loss
        cost = np.sum(correct_logprobs)/m
    return cost


# Compute the cost with regularization
# Input : Activation of last layer
#       : Output from data
#       : Y
#       :outputActivationFunc - Activation function at output - sigmoid/softmax
# Output: cost
def computeCostWithReg(parameters,AL,Y,lambd, outputActivationFunc="sigmoid"):
  

    if outputActivationFunc=="sigmoid":
        m= float(Y.shape[1])
        # Element wise multiply for logprobs
        cost=-1/m *np.sum(Y*np.log(AL) + (1-Y)*(np.log(1-AL)))
        cost = np.squeeze(cost)
        
        # Regularization cost
        L= int(len(parameters)/2) 
        L2RegularizationCost=0
        for l in range(L):
            L2RegularizationCost+=np.sum(np.square(parameters['W'+str(l+1)]))
        
        L2RegularizationCost = (lambd/(2*m))*L2RegularizationCost   
        cost = cost +  L2RegularizationCost

                        
    elif outputActivationFunc=="softmax":
        # Take transpose of Y for softmax
        Y=Y.T
        m= float(len(Y))
        # Compute log probs. Take the log prob of correct class based on output y
        correct_logprobs = -np.log(AL[range(int(m)),Y.T])
        # Conpute loss
        cost = np.sum(correct_logprobs)/m
        
               # Regularization cost
        L= int(len(parameters)/2) 
        L2RegularizationCost=0
        for l in range(L):
            L2RegularizationCost+=np.sum(np.square(parameters['W'+str(l+1)]))
        
        L2RegularizationCost = (lambd/(2*m))*L2RegularizationCost  
        cost = cost +  L2RegularizationCost
     
    return cost

# Compute the backpropoagation for 1 cycle
# Input : Neural Network parameters - dA
#       # cache - forward_cache & activation_cache
#       # Input features
#       # Output values Y
# Returns: Gradients
# dL/dWi= dL/dZi*Al-1
# dl/dbl = dL/dZl
# dL/dZ_prev=dL/dZl*W
def layerActivationBackward(dA, cache, Y, keep_prob=1, activationFunc="relu"):
    forward_cache, activation_cache = cache
    A_prev, W, b = forward_cache
    numtraining = float(A_prev.shape[1])
    #print("n=",numtraining)
    #print("no=",numtraining)
    if activationFunc == "relu":
        dZ = reluDerivative(dA, activation_cache)           
    elif activationFunc == "sigmoid":
        dZ = sigmoidDerivative(dA, activation_cache)      
    elif activationFunc == "tanh":
        dZ = tanhDerivative(dA, activation_cache)
    elif activationFunc == "softmax":
        dZ = stableSoftmaxDerivative(dA, activation_cache,Y,numtraining)
  
    if activationFunc == 'softmax':
        dW = 1/numtraining * np.dot(A_prev,dZ)
        db = 1/numtraining * np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ,W)
        

    else:
        #print(numtraining)
        dW = 1/numtraining *(np.dot(dZ,A_prev.T))
        #print("dW=",dW)
        db = 1/numtraining * np.sum(dZ, axis=1, keepdims=True)
        #print("db=",db)
        dA_prev = np.dot(W.T,dZ)    
              
    return dA_prev, dW, db


# Compute the backpropoagation with regularization for 1 cycle
# Input : Neural Network parameters - dA
#       # cache - forward_cache & activation_cache
#       # Input features
#       # Output values Y
# Returns: Gradients
# dL/dWi= dL/dZi*Al-1
# dl/dbl = dL/dZl
# dL/dZ_prev=dL/dZl*W
def layerActivationBackwardWithReg(dA, cache, Y, lambd, activationFunc):
    forward_cache, activation_cache = cache
    A_prev, W, b = forward_cache
    numtraining = float(A_prev.shape[1])
    #print("n=",numtraining)
    #print("no=",numtraining)
    if activationFunc == "relu":
        dZ = reluDerivative(dA, activation_cache)           
    elif activationFunc == "sigmoid":
        dZ = sigmoidDerivative(dA, activation_cache)      
    elif activationFunc == "tanh":
        dZ = tanhDerivative(dA, activation_cache)
    elif activationFunc == "softmax":
        dZ = stableSoftmaxDerivative(dA, activation_cache,Y,numtraining)
  
    if activationFunc == 'softmax':
        # Add the regularization factor
        dW = 1/numtraining * np.dot(A_prev,dZ) +  (lambd/numtraining) * W.T
        db = 1/numtraining * np.sum(dZ, axis=0, keepdims=True)
        dA_prev = np.dot(dZ,W)
    else:
        # Add the regularization factor
        dW = 1/numtraining *(np.dot(dZ,A_prev.T)) + (lambd/numtraining) * W
        #print("dW=",dW)
        db = 1/numtraining * np.sum(dZ, axis=1, keepdims=True)
        #print("db=",db)
        dA_prev = np.dot(W.T,dZ)    

        
    return dA_prev, dW, db

# Compute the backpropoagation for 1 cycle
# Input : AL: Output of L layer Network - weights
#       # Y  Real output
#       # caches -- list of caches containing:
#       every cache of layerActivationForward() with "relu"/"tanh"
#       #(it's caches[l], for l in range(L-1) i.e l = 0...L-2)
#       #the cache of layerActivationForward() with "sigmoid" (it's caches[L-1])
#       hiddenActivationFunc - Activation function at hidden layers - relu/sigmoid/tanh
#       #         outputActivationFunc - Activation function at output - sigmoid/softmax
#    
#   Returns:
#    gradients -- A dictionary with the gradients
#                 gradients["dA" + str(l)] = ... 
#                 gradients["dW" + str(l)] = ...

def backwardPropagationDeep(AL, Y, caches, dropoutMat, lambd=0, keep_prob=1, hiddenActivationFunc='relu',outputActivationFunc="sigmoid"):
    #initialize the gradients
    gradients = {}
    # Set the number of layers
    L = len(caches) 
    m = float(AL.shape[1])
    
    if outputActivationFunc == "sigmoid":
         Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL   
         # Initializing the backpropagation 
         # dl/dAL= -(y/a + (1-y)/(1-a)) - At the output layer
         dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  
    else:
        dAL =0
        Y=Y.T
    
    # Since this is a binary classification the activation at output is sigmoid
    # Get the gradients at the last layer
    # Inputs: "AL, Y, caches". 
    # Outputs: "gradients["dAL"], gradients["dWL"], gradients["dbL"]  
    current_cache = caches[L-1]
    if lambd==0:
       gradients["dA" + str(L)], gradients["dW" + str(L)], gradients["db" + str(L)] = layerActivationBackward(dAL, current_cache, 
                                            Y, activationFunc = outputActivationFunc)
    else: #Regularization
       gradients["dA" + str(L)], gradients["dW" + str(L)], gradients["db" + str(L)] = layerActivationBackwardWithReg(dAL, current_cache, 
                                            Y, lambd, activationFunc = outputActivationFunc)
    
    # Note dA for softmax is the transpose
    if outputActivationFunc == "softmax":
        gradients["dA" + str(L)] = gradients["dA" + str(L)].T
    # Traverse in the reverse direction
    for l in reversed(range(L-1)):
        # Compute the gradients for L-1 to 1 for Relu/tanh
        # Inputs: "gradients["dA" + str(l + 2)], caches". 
        # Outputs: "gradients["dA" + str(l + 1)] , gradients["dW" + str(l + 1)] , gradients["db" + str(l + 1)] 
        current_cache = caches[l]
       
        #dA_prev_temp, dW_temp, db_temp = layerActivationBackward(gradients['dA'+str(l+2)], current_cache, activationFunc = "relu")
        if lambd==0:
        
           # In the reverse direction use the dame dropout matrix
           # Random dropout
           # Multiply dA'l' with the dropoutMat and divide to keep the expected value same
           D = dropoutMat["D" + str(l+1)]          
           # Drop some dAl's
           gradients['dA'+str(l+2)]= np.multiply(gradients['dA'+str(l+2)],D)          
           # Divide by keep_prob to keep expected value same                                
           gradients['dA'+str(l+2)] = np.divide(gradients['dA'+str(l+2)],keep_prob) 
           
           dA_prev_temp, dW_temp, db_temp = layerActivationBackward(gradients['dA'+str(l+2)], current_cache, Y, keep_prob=1, activationFunc = hiddenActivationFunc) 
           
        else:
            dA_prev_temp, dW_temp, db_temp = layerActivationBackwardWithReg(gradients['dA'+str(l+2)], current_cache, Y, lambd, activationFunc = hiddenActivationFunc)
            
        gradients["dA" + str(l + 1)] = dA_prev_temp
        gradients["dW" + str(l + 1)] = dW_temp
        gradients["db" + str(l + 1)] = db_temp


    return gradients

# Perform Gradient Descent
# Input : Weights and biases
#       : gradients
#       : learning rate
#       : outputActivationFunc - Activation function at output - sigmoid/softmax
#output : Updated weights after 1 iteration
def gradientDescent(parameters, gradients, learningRate,outputActivationFunc="sigmoid"):
    
    L = int(len(parameters) / 2)
    # Update rule for each parameter. 
    for l in range(L-1):
        parameters["W" + str(l+1)] = parameters['W'+str(l+1)] -learningRate* gradients['dW' + str(l+1)] 
        parameters["b" + str(l+1)] = parameters['b'+str(l+1)] -learningRate* gradients['db' + str(l+1)]
    
    if outputActivationFunc=="sigmoid":
        parameters["W" + str(L)] = parameters['W'+str(L)] -learningRate* gradients['dW' + str(L)] 
        parameters["b" + str(L)] = parameters['b'+str(L)] -learningRate* gradients['db' + str(L)]
    elif outputActivationFunc=="softmax":
        parameters["W" + str(L)] = parameters['W'+str(L)] -learningRate* gradients['dW' + str(L)].T 
        parameters["b" + str(L)] = parameters['b'+str(L)] -learningRate* gradients['db' + str(L)].T

    
    
    return parameters





#  Execute a L layer Deep learning model
# Input : X - Input features
#       : Y output
#       : layersDimensions - Dimension of layers
#       : hiddenActivationFunc - Activation function at hidden layer relu /tanh/sigmoid
#       : learning rate
#       : num of iteration
#       : outputActivationFunc - Activation function at output - sigmoid/softmax
#output : Updated weights after 1 iteration
    
def L_Layer_DeepModel(X1, Y1, layersDimensions, hiddenActivationFunc='relu', outputActivationFunc="sigmoid", 
                      learningRate = .3,  lambd=0, keep_prob=1, num_iterations = 10000,initType="default", print_cost=False,figure="figa.png"):

    np.random.seed(1)
    costs = []                         
    
    # Parameters initialization.
    if initType == "He":
       parameters = HeInitializeDeepModel(layersDimensions)
    elif initType == "Xavier" :
       parameters = XavInitializeDeepModel(layersDimensions)
    else: #Default
       parameters = initializeDeepModel(layersDimensions)
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        AL, caches, dropoutMat = forwardPropagationDeep(X1, parameters, keep_prob, hiddenActivationFunc="relu",outputActivationFunc=outputActivationFunc)
        
        # Regularization parameter is 0
        if lambd==0:
            # Compute cost
            cost = computeCost(parameters,AL, Y1, outputActivationFunc=outputActivationFunc) 
        # Include L2 regularization
        else:
           # Compute cost
            cost = computeCostWithReg(parameters,AL, Y1, lambd, outputActivationFunc=outputActivationFunc) 

        # Backward propagation.      
        gradients = backwardPropagationDeep(AL, Y1, caches, dropoutMat, lambd, keep_prob, hiddenActivationFunc="relu",outputActivationFunc=outputActivationFunc)
         
        # Update parameters.
        parameters = gradientDescent(parameters, gradients, learningRate=learningRate,outputActivationFunc=outputActivationFunc)

                
        # Print the cost every 100 training example
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('No of iterations (x1000)')
    plt.title("Learning rate =" + str(learningRate))
    plt.savefig(figure,bbox_inches='tight')
    #plt.show()
    plt.clf()
    plt.close()


    
    return parameters

#  Execute a L layer Deep learning model Stoachastic Gradient Descent
# Input : X - Input features
#       : Y output
#       : layersDimensions - Dimension of layers
#       : hiddenActivationFunc - Activation function at hidden layer relu /tanh/sigmoid
#       : learning rate
#       : num of iteration
#       : outputActivationFunc - Activation function at output - sigmoid/softmax
#output : Updated weights after 1 iteration

def L_Layer_DeepModel_SGD(X1, Y1, layersDimensions, hiddenActivationFunc='relu', outputActivationFunc="sigmoid",learningRate = .3, mini_batch_size = 64, num_epochs = 2500, print_cost=False):#lr was 0.009

    np.random.seed(1)
    costs = []                         
    
    # Parameters initialization.
    parameters = initializeDeepModel(layersDimensions)
    seed=10
   # Loop for number of epochs
    for i in range(num_epochs):
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X1, Y1, mini_batch_size, seed)

        batch=0
        # Loop through each mini batch
        for minibatch in minibatches:
            #print("batch=",batch)
            batch=batch+1
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Perfrom forward propagation
            AL, caches = forwardPropagationDeep(minibatch_X, parameters,hiddenActivationFunc="relu",outputActivationFunc=outputActivationFunc)
        
            # Compute cost
            cost = computeCost(AL, minibatch_Y,outputActivationFunc=outputActivationFunc)
            #print("minibatch_Y=",minibatch_Y.shape)
            # Backward propagation.
            gradients = backwardPropagationDeep(AL, minibatch_Y, caches,hiddenActivationFunc="relu",outputActivationFunc=outputActivationFunc)
         
            # Update parameters.
            parameters = gradientDescent(parameters, gradients, learningRate=learningRate,outputActivationFunc=outputActivationFunc)                
        
        # Print the cost every 1000 epoch
        if print_cost and i % 100 == 0:
            print ("Cost after epoch %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('No of iterations')
    plt.title("Learning rate =" + str(learningRate))
    #plt.show()
    plt.savefig("fig1",bbox_inches='tight')
    plt.close()
    return parameters


# Create random mini batches
def random_mini_batches(X, Y, miniBatchSize = 64, seed = 0):
    
    np.random.seed(seed)    
    # Get number of training samples       
    m = X.shape[1] 
    # Initialize mini batches     
    mini_batches = []
        
    # Create  a list of random numbers < m
    permutation = list(np.random.permutation(m))
    # Randomly shuffle the training data
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # Compute number of mini batches
    numCompleteMinibatches = math.floor(m/miniBatchSize)

   # For the number of mini batches
    for k in range(0, numCompleteMinibatches):

        # Set the start and end of each mini batch
        mini_batch_X = shuffled_X[:, k*miniBatchSize : (k+1) * miniBatchSize]
        mini_batch_Y = shuffled_Y[:, k*miniBatchSize : (k+1) * miniBatchSize]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    

    #if m % miniBatchSize != 0:. The batch does not evenly divide by the mini batch
    if m % miniBatchSize != 0:
        l=math.floor(m/miniBatchSize)*miniBatchSize
        # Set the start and end of last mini batch
        m=l+m % miniBatchSize
        mini_batch_X = shuffled_X[:,l:m]
        mini_batch_Y = shuffled_Y[:,l:m]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

# Plot a decision boundary
# Input : Input Model,
#         X
#         Y
#         sz - Num of hiden units
#         lr - Learning rate
#         Fig to be saved as
# Returns Null
def plot_decision_boundary(model, X, y,lr,figure1="figb.png"):
    print("plot")
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1   
    colors=['black','gold']
    cmap = matplotlib.colors.ListedColormap(colors)   
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap="coolwarm")
    plt.ylabel('x2')
    plt.xlabel('x1')
    x=X.T
    y=y.T.reshape(300,)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=20);
    print(X.shape)
    plt.title("Decision Boundary for learning rate:"+lr)
    plt.savefig(figure1, bbox_inches='tight')
    #plt.show()

    
def predict(parameters, X,keep_prob=1,hiddenActivationFunc="relu",outputActivationFunc="sigmoid"):
    A2, cache,dropoutMat = forwardPropagationDeep(X, parameters, keep_prob=1, hiddenActivationFunc="relu",outputActivationFunc=outputActivationFunc)
    predictions = (A2>0.5)    
    return predictions

def predict_proba(parameters, X,outputActivationFunc="sigmoid"):
    A2, cache = forwardPropagationDeep(X, parameters)
    if outputActivationFunc=="sigmoid":
       proba=A2  
    elif outputActivationFunc=="softmax":
       proba=np.argmax(A2, axis=0).reshape(-1,1)
       print("A2=",A2.shape)
    return proba

# Plot a decision boundary
# Input : Input Model,
#         X
#         Y
#         sz - Num of hiden units
#         lr - Learning rate
#         Fig to be saved as
# Returns Null
def plot_decision_boundary1(X, y,W1,b1,W2,b2,figure2="figc.png"):
    #plot_decision_boundary(lambda x: predict(parameters, x.T), x1,y1.T,str(0.3),"fig2.png") 
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W1.T) + b1.T), W2.T) + b2.T
    Z = np.argmax(Z, axis=1)
    Z = Z.reshape(xx.shape)
    
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    print(X.shape)
    y1=y.reshape(300,)
    plt.scatter(X[:, 0], X[:, 1], c=y1, s=40, cmap=plt.cm.Spectral)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.savefig(figure2, bbox_inches='tight')
    
    
def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    print(train_X.shape)
    print(train_Y.shape)
    print("load")
    #plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y