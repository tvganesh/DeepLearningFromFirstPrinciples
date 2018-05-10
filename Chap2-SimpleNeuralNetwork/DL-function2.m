1;
# Define sigmoid function
function a = sigmoid(z)
  a = 1 ./ (1+ exp(-z));
end

# Compute the loss
function loss=computeLoss(numtraining,Y,A)
  loss = -1/numtraining * sum((Y .* log(A)) + (1-Y) .* log(1-A));
end
  
# Compute the model shape given the dataset
function [n_x,m,n_h,n_y] = getModelShape(X,Y)
    m= size(X)(2);
    n_x=size(X)(1);
    n_h=4;
    n_y=size(Y)(1);
end

function [W1,b1,W2,b2] = modelInit(n_x,n_h,n_y)
    rand ("seed", 2);
    W1=rand(n_h,n_x)*0.01; # Set the initial values to a small number
    b1=zeros(n_h,1);
    W2=rand(n_y,n_h)*0.01;
    b2=zeros(n_y,1);

end

# Compute the forward propoagation through the neural network
# Input : Features
#         Weight and bias matrices and vectors
# Returns : The Activation of 2nd layer
#         : Output and activation of layer 1 & 2
function [Z1,A1,Z2,A2]= forwardPropagation(X,W1,b1,W2,b2)
    # Get the parameters

    # Determine the number of training samples
    m=size(X)(2);
    # Compute Z1 of the input layer
    # Octave also handles broadcasting like Python!!
    Z1=W1 * X +b1;
    # Compute the output A1 with the tanh activation function. The tanh activation function
    # performs better than the sigmoid function
    A1=tanh(Z1);
    
    # Compute Z2 of the input layer
    Z2=W2 * A1+b2;
    # Compute the output A1 with the tanh activation function. The tanh activation function
    # performs better than the sigmoid function
    A2=sigmoid(Z2);    
   
end

# Compute the cost
# Input : Activation of 2nd layer
#       : Output from data
# Output: cost
function [cost] = computeCost(A,Y)
    numTraining= size(Y)(2);
    # Element wise multiply for logprobs
    cost = -1/numTraining * sum((Y .* log(A)) + (1-Y) .* log(1-A));
end


# Compute the backpropoagation for 1 cycle
# Input : Neural Network parameters - weights and biases
#       # Z and Activations of 2 layers
#       # Input features
#       # Output values Y
# Returns: Gradients
function [dW1,db1,dW2,db2]= backPropagation(W1,W2,A1,A2, X, Y)
    numTraining=size(X)(2);
 
    dZ2 = A2 - Y;
    dW2 = 1/numTraining * dZ2 * A1';
    db2 = 1/numTraining * sum(dZ2);
    
    dZ1 =  W2' * dZ2 .* (1 - power(A1, 2));
    dW1 = 1/numTraining *  dZ1 * X';
    # Note the '2' in the next statement indicates that a row sum has to done , 2nd dimension
    db1 = 1/numTraining * sum(dZ1,2);

end

# Perform Gradient Descent
# Input : Weights and biases
#       : gradients
#       : learning rate
#output : Updated weights after 1 iteration
function [W1,b1,W2,b2]= gradientDescent(W1,b1,W2,b2, dW1,db1,dW2,db2, learningRate)
    W1 = W1-learningRate*dW1;
    b1 = b1-learningRate*db1;
    W2 = W2-learningRate*dW2;
    b2 = b2-learningRate*db2;
end


# Compute the Neural Network  by minimizing the cost 
# Input : Input data X,
#         Output Y
#         No of hidden units in hidden layer
#         No of iterations
# Returns  Updated weight and bias vectors of the neural network
function [W1,b1,W2,b2,costs]= computeNN(X, Y,numHidden, learningRate, numIterations = 10000)

    [numFeats,numTraining,n_h,numOutput] = getModelShape(X, Y);
 
    costs=[];

    [W1,b1,W2,b2] = modelInit(numFeats,numHidden,numOutput) ;
    #W1 =[-0.00416758, -0.00056267; -0.02136196,  0.01640271; -0.01793436, -0.00841747;  0.00502881 -0.01245288];
    #W2=[-0.01057952, -0.00909008,  0.00551454,  0.02292208];
    #b1=[0;0;0;0];
    #b2=[0];
    # Perform gradient descent
    for i =0:numIterations
        # Evaluate forward prop to compute activation at output layer
        [Z1,A1,Z2,A2] =  forwardPropagation(X, W1,b1,W2,b2);      
        # Compute cost from Activation at output and Y
        cost = computeCost(A2, Y);
        # Perform backprop to compute gradients
        [dW1,db1,dW2,db2] = backPropagation(W1,W2,A1,A2, X, Y); 
        # Use gradients to update the weights for each iteration.

        [W1,b1,W2,b2] = gradientDescent(W1,b1,W2,b2, dW1,db1,dW2,db2,learningRate);     
        # Print the cost every 1000 iterations
        if ( mod(i,1000) == 0)
            costs =[costs cost];
            #disp ("Cost after iteration"), disp(i),disp(cost);
        endif
     endfor
 end
 
 


# Compute the predicted value for a given input
# Input : Neural Network parameters
#       : Input data
function [predictions]= predict(W1,b1,W2,b2, X)
    [Z1,A1,Z2,A2] = forwardPropagation(X, W1,b1,W2,b2);
    predictions = (A2>0.5);
end 

# Plot the decision boundary
function plotDecisionBoundary(data,W1,b1,W2,b2)
    %Plot a non-linear decision boundary learned by the SVM
   colormap ("default");

    % Make classification predictions over a grid of values
    x1plot = linspace(min(data(:,1)), max(data(:,1)), 400)';
    x2plot = linspace(min(data(:,2)), max(data(:,2)), 400)';
    [X1, X2] = meshgrid(x1plot, x2plot);
    vals = zeros(size(X1));
    # Plot the prediction for the grid
    for i = 1:size(X1, 2)
       gridPoints = [X1(:, i), X2(:, i)];
       vals(:, i)=predict(W1,b1,W2,b2,gridPoints');
    endfor
   
    scatter(data(:,1),data(:,2),8,c=data(:,3),"filled");
    % Plot the boundary
    hold on
    #contour(X1, X2, vals, [0 0], 'LineWidth', 2);
    contour(X1, X2, vals);
    title ({"3 layer Neural Network decision boundary"});
    hold off;

end

# Plot the cost vs iterations
function plotLRCostVsIterations()
    data=csvread("data.csv");

    X=data(:,1:2);
    Y=data(:,3);
    lr=[0.5,1.2,3]
    col='kbm'
    for i=1:3
       [W1,b1,W2,b2,costs]= computeNN(X', Y',4, learningRate=lr(i), numIterations = 10000);
       iterations = 1000*[0:10];
       hold on;
       plot(iterations,costs,color=col(i),"linewidth", 3);
       hold off;
       title ("Cost vs no of iterations for different learning rates");
       xlabel("No of iterations")
       ylabel("Cost")
       legend('0.5','1.2','3.0')
    endfor
end

# Plot the cost vs number of hidden units
function plotHiddenCostVsIterations()
    data=csvread("data1.csv");

    X=data(:,1:2);
    Y=data(:,3);
    hidden=[4,9,12]
    col='kbm'
    for i=1:3
       [W1,b1,W2,b2,costs]= computeNN(X', Y',hidden(i), learningRate=1.5, numIterations = 10000);
       iterations = 1000*[0:10];
       hold on;
       plot(iterations,costs,color=col(i),"linewidth", 3);
       hold off;
       title ("Cost vs no of iterations for different number of hidden units");
       xlabel("No of iterations")
       ylabel("Cost")
       legend('4','9','12')
    endfor
end