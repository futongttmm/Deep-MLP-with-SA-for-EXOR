import numpy as np
from MLPmodel import L_layer_model, predict

train_x = np.array([[0, 1, 1], [0, 0, 1]])
train_y = np.array([1, 0, 0]).reshape(1, 3)
test_x = np.array([[0], [0]])

test_y = np.array([1]).reshape(1, 1)

layers_dims = (2, 2, 1)


annealing = int(input('Annealing or not? Please use level 1, 10, or 100   '))
'''
try:
    if annealing != 1 or annealing != 10 or annealing != 100:
        raise NameError('Annealing level: ' + str(annealing))

except NameError:
    print('Input an unsolvalbe annealing level! Please use level 1, 10, or 100')
'''
parameters = L_layer_model(train_x, train_y, layers_dims, annealing=annealing, num_iterations = 100000, print_cost = True, plot=True)
print(parameters)

pred_train = predict(train_x, train_y, parameters, annealing=1)
pred_test = predict(test_x, test_y, parameters, annealing=1)