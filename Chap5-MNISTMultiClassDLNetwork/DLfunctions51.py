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

# Compute the activation at a layer 'l' for forward prop in a Deep Network
# Input : A_prec - Activation of previous layer
#         W,b - Weight and bias matrices and vectors
#         activationFunc - Activation function - sigmoid, tanh, relu etc
# Returns : The Activation of this layer
#         : 
# Z = W * X + b
# A = sigmoid(Z), A= Relu(Z), A= tanh(Z)
def layerActivationForward(A_prev, W, b, activationFunc):
    
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
def forwardPropagationDeep(X, parameters,hiddenActivationFunc='relu',outputActivationFunc='sigmoid'):
    caches = []
    # Set A to X (A0)
    A = X
    L = int(len(parameters)/2) # number of layers in the neural network
    # Loop through from layer 1 to upto layer L
    for l in range(1, L):
        A_prev = A 
        # Zi = Wi x Ai-1 + bi  and Ai = g(Zi)
        #A, cache = layerActivationForward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activationFunc = "relu")
        A, cache = layerActivationForward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activationFunc = hiddenActivationFunc)
        caches.append(cache)
        #print("l=",l)
        #print(A)
    
    # Since this is binary classification use the sigmoid activation function in
    # last layer   
    AL, cache = layerActivationForward(A, parameters['W'+str(L)], parameters['b'+str(L)], activationFunc = outputActivationFunc)
    caches.append(cache)
            
    return AL, caches


# Compute the cost
# Input : Activation of last layer
#       : Output from data
#       : Y
#       :outputActivationFunc - Activation function at output - sigmoid/softmax
# Output: cost
def computeCost(AL,Y,outputActivationFunc="sigmoid"):

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

# Compute the backpropoagation for 1 cycle
# Input : Neural Network parameters - dA
#       # cache - forward_cache & activation_cache
#       # Input features
#       # Output values Y
# Returns: Gradients
# dL/dWi= dL/dZi*Al-1
# dl/dbl = dL/dZl
# dL/dZ_prev=dL/dZl*W
def layerActivationBackward(dA, cache, Y, activationFunc):
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

def backwardPropagationDeep(AL, Y, caches,hiddenActivationFunc='relu',outputActivationFunc="sigmoid"):
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
    gradients["dA" + str(L)], gradients["dW" + str(L)], gradients["db" + str(L)] = layerActivationBackward(dAL, current_cache, Y, activationFunc = outputActivationFunc)
    
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
        dA_prev_temp, dW_temp, db_temp = layerActivationBackward(gradients['dA'+str(l+2)], current_cache, Y, activationFunc = hiddenActivationFunc)
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
    
def L_Layer_DeepModel(X1, Y1, layersDimensions, hiddenActivationFunc='relu', outputActivationFunc="sigmoid", learningRate = .3, num_iterations = 10000, print_cost=False):#lr was 0.009

    np.random.seed(1)
    costs = []                         
    
    # Parameters initialization.
    parameters = initializeDeepModel(layersDimensions)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        #AL, caches = forwardPropagationDeep(X, parameters,hiddenActivationFunc)
        
        # Compute cost.
        #cost = computeCost(AL, Y)
   
        # Backward propagation.
        #gradients = backwardPropagationDeep(AL, Y, caches,hiddenActivationFunc)
 
        ## Update parameters.
        #parameters = gradientDescent(parameters, gradients, learning_rate)

        AL, caches = forwardPropagationDeep(X1, parameters,hiddenActivationFunc="relu",outputActivationFunc=outputActivationFunc)
        
        # Compute cost
        cost = computeCost(AL, Y1,outputActivationFunc=outputActivationFunc)
        #print("Y1=",Y1.shape)
        # Backward propagation.
        gradients = backwardPropagationDeep(AL, Y1, caches,hiddenActivationFunc="relu",outputActivationFunc=outputActivationFunc)
         
        # Update parameters.
        parameters = gradientDescent(parameters, gradients, learningRate=learningRate,outputActivationFunc=outputActivationFunc)

                
        # Print the cost every 100 training example
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('No of iterations (x100)')
    plt.title("Learning rate =" + str(learningRate))
    #plt.show()
    plt.savefig("fig1",bbox_inches='tight')
    
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
def plot_decision_boundary(model, X, y,lr,fig):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1   
    colors=['black','yellow']
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
    plt.scatter(X[0, :], X[1, :], c=y, s=7,cmap=cmap)
    plt.title("Decision Boundary for learning rate:"+lr)
    #plt.show()
    plt.savefig(fig, bbox_inches='tight')
    
def predict(parameters, X):
    A2, cache = forwardPropagationDeep(X, parameters)
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
def plot_decision_surface(model, X, y,sz,lr,fig):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1   
    z_min, z_max = X[2, :].min() - 1, X[2, :].max() + 1   
    colors=['black','gold']
    cmap = matplotlib.colors.ListedColormap(colors)   
    h = 3
    # Generate a grid of points with distance h between them
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(z_min, z_max, h))
    # Predict the function value for the whole grid
    a=np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    
    Z = predict(parameters,a.T)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    #plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(xx, yy, Z, 50, cmap='binary')
    #plt.ylabel('x2')
    #plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=cmap)
    plt.title("Decision Boundary for hidden layer size:" + sz +" and learning rate:"+lr)
    plt.show()
    
def plotSurface(X,parameters):
    
    #xx, yy, zz = np.meshgrid(np.arange(10), np.arange(10), np.arange(10))
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1   
    z_min, z_max = X[2, :].min() - 1, X[2, :].max() + 1   
    colors=['red']
    cmap = matplotlib.colors.ListedColormap(colors)   
    h = 1
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), 
                             np.arange(z_min, z_max, h))
    # For the meh grid values predict a model
    a=np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    Z = predict(parameters,a.T)
    r=Z.T
    r1=r.reshape(xx.shape)
    # Find teh values for which the repdiction is 1
    xx1=xx[r1]
    yy1=yy[r1]
    zz1=zz[r1]
    # Plot these values
    ax = plt.axes(projection='3d')
    #ax.plot_trisurf(xx1, yy1, zz1, cmap='bone', edgecolor='none');
    ax.scatter3D(xx1, yy1,zz1, c=zz1,s=10,cmap=cmap)
    #ax.plot_surface(xx1, yy1, zz1, 'gray')