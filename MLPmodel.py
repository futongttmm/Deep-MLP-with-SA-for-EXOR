import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits import mplot3d
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from Activation import relu, sigmoid
from MLP import (L_model_backward, L_model_forward, compute_cost,
                 initialize_parameters_deep, update_parameters)


def plot_annealing(parameters, annealing):
    # plotting in 3D
        fig, ax = plt.subplots(1, 2, figsize=(16, 6), subplot_kw={'projection': '3d'})

        # Plot individual data points in this sub-figure                
        ax[0].scatter(0, 0, c='r', label="Class 0")
        ax[0].scatter(0, 1, c='r', label="Class 0")
        ax[0].scatter(1, 0, c='r', label="Class 0")
        ax[0].scatter(1, 1, c='b', label="Class 1")
        ax[1].scatter(0, 0, c='r', label="Class 0")
        ax[1].scatter(0, 1, c='r', label="Class 0")
        ax[1].scatter(1, 0, c='r', label="Class 0")
        ax[1].scatter(1, 1, c='b', label="Class 1")
                
        x_1_analog = np.arange(0, 1, 0.1)
        x_2_analog = np.arange(0, 1, 0.1)
                
        # we need a mesh-grid for 3-Dimensional plotting
        X_1_analog, X_2_analog = np.meshgrid(x_1_analog, x_2_analog)
        
        X_1_ = X_1_analog.reshape(1, len(X_1_analog) * len(X_1_analog[0,:]))
        X_2_ = X_2_analog.reshape(1, len(X_2_analog) * len(X_2_analog[0,:]))
        X_analog = np.vstack((X_1_, X_2_))

        AL, _ = L_model_forward(X_analog, parameters, annealing)
        
        # Set true values
        Y_analog = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                            0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        # Compute errors (avoid divided by zero)
        E = -(np.multiply(np.log(AL), Y_analog) + np.multiply(np.log(1-AL), 1-Y_analog))

        AL = AL.reshape(len(X_1_analog), len(X_2_analog))
        E = E.reshape(len(X_1_analog), len(X_2_analog))
        
        ax[0].set_title('The Hyper-plane after Applying Sigmoid()  ' + str(annealing))     
        surf1 = ax[0].plot_surface(X_1_analog, X_2_analog, AL,cmap=cm.coolwarm,
                           linewidth=0, antialiased=False )        
        
        ax[1].set_title('The Hyper-plane for Loss') 
        surf2 = ax[1].plot_surface(X_1_analog, X_2_analog, E,  cmap='viridis', edgecolor='none') 

        # Add a color bar which maps values to colors.
        fig.colorbar(surf1, ax=ax[0], shrink=0.5)
        fig.colorbar(surf2, ax=ax[1], shrink=0.5)

        plt.tight_layout()
        plt.show()


def L_layer_model(X, Y, layers_dims, annealing, learning_rate = 0.0075, num_iterations = 3000, print_cost=False, plot=False):
    """
    X -- data -- (num_px * num_px * 3, number of examples)
    Y -- true desired vector, of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed()
    costs = []  # keep track of cost
    
    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)
    # print(parameters)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters, annealing)
        
        # Compute cost.
        cost = compute_cost(AL, Y)
    
        # Backward propagation.
        grads =  L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Plot 'annealing'.
        if plot == True and i % 99999 == 0 and i != 0:
            plot_annealing(parameters, annealing)
                
        # Print the cost every 1000 training example.
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        costs.append(cost)  
            
    # plot the cost
    '''
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    '''

    return parameters


def predict(X, y, parameters, annealing):
    """
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, _ = L_model_forward(X, parameters, annealing)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    print ("predictions: " + str(p))
    print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p
