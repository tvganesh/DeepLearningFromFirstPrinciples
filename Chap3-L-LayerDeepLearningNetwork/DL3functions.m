1;
# Define sigmoid function
function [A,cache] = sigmoid(Z)
  A = 1 ./ (1+ exp(-Z));
  cache=Z;
end

# Define Relu function
function [A,cache] = relu(Z)
  A = max(0,Z);
  cache=Z;
end

# Define Relu function
function [A,cache] = tanhAct(Z)
  A = tanh(Z);
  cache=Z;
end

# Define Relu Derivative 
function [dZ] = reluDerivative(dA,cache)
  Z = cache;
  dZ = dA;
  # Get elements that are greater than 0
  a = (Z > 0);
  # Select only those elements where Z > 0
  dZ = dZ .* a;
end

# Define Sigmoid Derivative 
function [dZ] = sigmoidDerivative(dA,cache)
  Z = cache;
  s = 1 ./ (1+ exp(-Z));
  dZ = dA .* s .* (1-s);
end

# Define Tanh Derivative 
function [dZ] = tanhDerivative(dA,cache)
  Z = cache;
  a = tanh(Z);
  dZ = dA .* (1 - a .^ 2);
end

# Initialize the model 
# Input : number of features
#         number of hidden units
#         number of units in output
# Returns: Weight and bias matrices and vectors


# Initialize model for L layers
# Input : List of units in each layer
# Returns: Initial weights and biases matrices for all layers
function [W b] = initializeDeepModel(layerDimensions)
    rand ("seed", 3);
    # note the Weight matrix at layer 'l' is a matrix of size (l,l-1)
    # The Bias is a vectors of size (l,1)
    
    # Loop through the layer dimension from 1.. L
    # Create cell arrays for Weights and biases

    for l =2:size(layerDimensions)(2)
         W{l-1} = rand(layerDimensions(l),layerDimensions(l-1))*0.01; #  Multiply by .01 
         b{l-1} = zeros(layerDimensions(l),1);       
   
    endfor
end

# Compute the activation at a layer 'l' for forward prop in a Deep Network
# Input : A_prec - Activation of previous layer
#         W,b - Weight and bias matrices and vectors
#         activationFunc - Activation function - sigmoid, tanh, relu etc
# Returns : The Activation of this layer
#         : 
# Z = W * X + b
# A = sigmoid(Z), A= Relu(Z), A= tanh(Z)
function [A forward_cache activation_cache] = layerActivationForward(A_prev, W, b, activationFunc)
    
    # Compute Z
    Z = W * A_prev +b;
    # Create a cell array
    forward_cache = {A_prev  W  b};
    # Compute the activation for sigmoid
    if (strcmp(activationFunc,"sigmoid"))
        [A activation_cache] = sigmoid(Z); 
    elseif (strcmp(activationFunc, "relu"))  # Compute the activation for Relu
        [A activation_cache] = relu(Z);
    elseif(strcmp(activationFunc,'tanh'))     # Compute the activation for tanh
        [A activation_cache] = tanhAct(Z);
    endif

end

# Compute the forward propagation for layers 1..L
# Input : X - Input Features
#         paramaters: Weights and biases
#         hiddenActivationFunc - Activation function at hidden layers Relu/tanh
# Returns : AL 
#           caches
# The forward propoagtion uses the Relu/tanh activation from layer 1..L-1 and sigmoid actiovation at layer L
function [AL forward_caches activation_caches] = forwardPropagationDeep(X, weights,biases, hiddenActivationFunc='relu')
    # Create an empty cell array
    forward_caches = {};
    activation_caches = {};
    # Set A to X (A0)
    A = X;
    L = length(weights); # number of layers in the neural network
    # Loop through from layer 1 to upto layer L
    for l =1:L-1
        A_prev = A; 
        # Zi = Wi x Ai-1 + bi  and Ai = g(Zi)
        W = weights{l};
        b = biases{l};
        [A forward_cache activation_cache] = layerActivationForward(A_prev, W,b, activationFunc=hiddenActivationFunc);
        forward_caches{l}=forward_cache;
        activation_caches{l} = activation_cache;
    endfor
    # Since this is binary classification use the sigmoid activation function in
    # last layer   
    W = weights{L};
    b = biases{L};
    [AL, forward_cache activation_cache] = layerActivationForward(A, W,b, activationFunc = "sigmoid");
    forward_caches{L}=forward_cache;
    activation_caches{L} = activation_cache;
            
end

# Compute the cost
# Input : Activation of last layer
#       : Output from data
# Output: cost
function [cost]= computeCost(AL,Y)
    numTraining= size(Y)(2);
    # Element wise multiply for logprobs
    cost = -1/numTraining * sum((Y .* log(AL)) + (1-Y) .* log(1-AL));
end

# Compute the backpropoagation for 1 cycle
# Input : Neural Network parameters - dA
#       # cache - forward_cache & activation_cache
#       # Input features
#       # Output values Y
# Returns: Gradients
# dL/dWi= dL/dZi*Al-1
# dl/dbl = dL/dZl
# dL/dZ_prev=dL/dZl*W
function [dA_prev dW db] =  layerActivationBackward(dA, forward_cache, activation_cache, activationFunc)

    if (strcmp(activationFunc,"relu"))
        dZ = reluDerivative(dA, activation_cache);           
    elseif (strcmp(activationFunc,"sigmoid"))
        dZ = sigmoidDerivative(dA, activation_cache);      
    elseif(strcmp(activationFunc, "tanh"))
        dZ = tanhDerivative(dA, activation_cache);
    endif
    A_prev = forward_cache{1};
    W =forward_cache{2};
    b = forward_cache{3};
    numTraining = size(A_prev)(2);
    dW = 1/numTraining * dZ * A_prev';
    db = 1/numTraining * sum(dZ,2);
    dA_prev = W'*dZ;
        
end 


# Compute the backpropoagation for 1 cycle
# Input : AL: Output of L layer Network - weights
#       # Y  Real output
#       # caches -- list of caches containing:
#       every cache of layerActivationForward() with "relu"/"tanh"
#       #(it's caches[l], for l in range(L-1) i.e l = 0...L-2)
#       #the cache of layerActivationForward() with "sigmoid" (it's caches[L-1])
#       hiddenActivationFunc - Activation function at hidden layers
#    
#   Returns:
#    gradients -- A dictionary with the gradients
#                 gradients["dA" + str(l)] = ... 
#                 gradients["dW" + str(l)] = ...

function [gradsDA gradsDW gradsDB]= backwardPropagationDeep(AL, Y, activation_caches,forward_caches,hiddenActivationFunc='relu')
    

    # Set the number of layers
    L = length(activation_caches); 
    m = size(AL)(2);

    
    # Initializing the backpropagation 
    # dl/dAL= -(y/a + (1-y)/(1-a)) - At the output layer
    dAL = -((Y ./ AL) - (1 - Y) ./ ( 1 - AL));    
    
    # Since this is a binary classification the activation at output is sigmoid
    # Get the gradients at the last layer
    # Inputs: "AL, Y, caches". 
    # Outputs: "gradients["dAL"], gradients["dWL"], gradients["dbL"]  
    activation_cache = activation_caches{L};
    forward_cache = forward_caches(L);
    # Note the cell array includes an array of forward caches. To get to this we need to include the index {1}
    [dA dW db] = layerActivationBackward(dAL, forward_cache{1}, activation_cache, activationFunc = "sigmoid");
    gradsDA{L}= dA;
    gradsDW{L}= dW;
    gradsDB{L}= db;

    # Traverse in the reverse direction
    for l =(L-1):-1:1
        # Compute the gradients for L-1 to 1 for Relu/tanh
        # Inputs: "gradients["dA" + str(l + 2)], caches". 
        # Outputs: "gradients["dA" + str(l + 1)] , gradients["dW" + str(l + 1)] , gradients["db" + str(l + 1)] 
        activation_cache = activation_caches{l};
        forward_cache = forward_caches(l);
       
        #dA_prev_temp, dW_temp, db_temp = layerActivationBackward(gradients['dA'+str(l+1)], current_cache, activationFunc = "relu")
        # dAl the dervative of the activation of the lth layer,is the first element
        dAl= gradsDA{l+1};
        [dA_prev_temp, dW_temp, db_temp] = layerActivationBackward(dAl, forward_cache{1}, activation_cache,  activationFunc = hiddenActivationFunc);
        gradsDA{l}= dA_prev_temp;
        gradsDW{l}= dW_temp;
        gradsDB{l}= db_temp;

    endfor

end


# Perform Gradient Descent
# Input : Weights and biases
#       : gradients
#       : learning rate
#output : Updated weights after 1 iteration
function [weights biases] = gradientDescent(weights, biases,gradsW,gradsB, learningRate)

    L = size(weights)(2); # number of layers in the neural network

    # Update rule for each parameter. 
    for l=1:L
        weights{l} = weights{l} -learningRate* gradsW{l}; 
        biases{l} = biases{l} -learningRate* gradsB{l};
    endfor
end



function [weights biases costs] = L_Layer_DeepModel(X, Y, layersDimensions, hiddenActivationFunc='relu', learning_rate = .3, num_iterations = 10000)#lr was 0.009

    rand ("seed", 1);
    costs = [] ;                        
    
    # Parameters initialization.
    [weights biases] = initializeDeepModel(layersDimensions);
    
    # Loop (gradient descent)
    for i = 0:num_iterations
        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        [AL forward_caches activation_caches] = forwardPropagationDeep(X, weights, biases,hiddenActivationFunc);
        
        # Compute cost.
        cost = computeCost(AL, Y);
   
        # Backward propagation.
        [gradsDA gradsDW gradsDB] = backwardPropagationDeep(AL, Y, activation_caches,forward_caches,hiddenActivationFunc);
 
        # Update parameters.
        [weights biases] = gradientDescent(weights,biases, gradsDW,gradsDB,learning_rate);

                
          # Print the cost every 1000 iterations
        if ( mod(i,1000) == 0)
            costs =[costs cost];
            #disp ("Cost after iteration"), disp(i),disp(cost);
            printf("Cost after iteration i=%i cost=%d\n",i,cost);
        endif
     endfor
    
end
 
 
 function plotCostVsIterations(maxIterations,costs)
     iterations=[0:1000:maxIterations];
     plot(iterations,costs);
     title ("Cost vs no of iterations for different learning rates");
     xlabel("No of iterations");
     ylabel("Cost");
end;

# Compute the predicted value for a given input
# Input : Neural Network parameters
#       : Input data
function [predictions]= predict(weights, biases, X,hiddenActivationFunc="relu")
    [AL forward_caches activation_caches] = forwardPropagationDeep(X, weights, biases,hiddenActivationFunc);
    predictions = (AL>0.5);
end 

# Plot the decision boundary
function plotDecisionBoundary(data,weights, biases,hiddenActivationFunc="relu")
    %Plot a non-linear decision boundary learned by the SVM
   colormap ("summer");

    % Make classification predictions over a grid of values
    x1plot = linspace(min(data(:,1)), max(data(:,1)), 400)';
    x2plot = linspace(min(data(:,2)), max(data(:,2)), 400)';
    [X1, X2] = meshgrid(x1plot, x2plot);
    vals = zeros(size(X1));
    # Plot the prediction for the grid
    for i = 1:size(X1, 2)
       gridPoints = [X1(:, i), X2(:, i)];
       vals(:, i)=predict(weights, biases,gridPoints',hiddenActivationFunc=hiddenActivationFunc);
    endfor
   
    scatter(data(:,1),data(:,2),8,c=data(:,3),"filled");
    % Plot the boundary
    hold on
    #contour(X1, X2, vals, [0 0], 'LineWidth', 2);
    contour(X1, X2, vals,"linewidth",4);
    title ({"3 layer Neural Network decision boundary"});
    hold off;

end

function [AL]= scores(weights, biases, X,hiddenActivationFunc="relu")
    [AL forward_caches activation_caches] = forwardPropagationDeep(X, weights, biases,hiddenActivationFunc);
end 


