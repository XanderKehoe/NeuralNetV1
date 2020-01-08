import random
import multiprocessing as mp
import math
from copy import deepcopy

import numpy as np

processor_use_count = 23

weight_list = []  # for debugging only
learning_rate = .2


class NeuralNet:
    # nn_structure is an array where each value specifies the size of each node (input, hidden, and output) layer.
    def __init__(self, nn_structure):
        self.activated_nodes = [0 for col in range(len(nn_structure))]
        for i in range(len(self.activated_nodes)):
            self.activated_nodes[i] = [0 for row in range(nn_structure[i])]
        self.inactive_nodes = self.activated_nodes

        self.weights = []
        for i in range(len(self.activated_nodes) - 1):
            self.weights.append([])
            for j in range(len(self.activated_nodes[i + 1])):
                self.weights[i].append([])
                for k in range(len(self.activated_nodes[i])):
                    self.weights[i][j].append(0)

    def randomize_weights(self, scalar):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] = scalar * ((2 * random.random()) - 1)  # generated random # between -1 and 1 times the scalar

    def print_nodes(self):
        for i in self.activated_nodes:
            print(i, end=" ")
        print("\n")

    def print_weights(self):
        for i in range(len(self.weights)):
            print("i:" + str(i))
            for j in range(len(self.weights[i])):
                print("\tj:" + str(j))
                for k in range(len(self.weights[i][j])):
                    print("\t\tk: " + str(k) + " -> " + str(self.weights[i][j][k]))
        print()

    def feed_forward(self, input_layer):
        if len(input_layer) == len(self.activated_nodes[0]):
            self.activated_nodes[0] = input_layer
            for i in range(len(self.activated_nodes) - 1):
                for j in range(len(self.activated_nodes[i + 1])):
                    #  tally up sum of weights * inputs to feed forward to next node layer
                    sum = 0;
                    for k in range(len(self.activated_nodes[i])):
                        sum += self.activated_nodes[i][k] * self.weights[i][j][k]
                    self.inactive_nodes[i + 1][j] = sum
                    self.activated_nodes[i + 1][j] = self.activation_function(self.inactive_nodes[i + 1][j], False)

            return self.activated_nodes[len(self.activated_nodes) - 1]
        else:
            print("FEED FORWARD ERROR: input and input layer sizes do not match!")

    @staticmethod
    def activation_function(x, derivative):
        if not derivative:
            return 1/(1+np.exp(-x))  # sigmoid
        else:
            return np.exp(-x)/(math.pow(1+np.exp(-x), 2))  # sigmoid

    def cost_sum(self, target_layer):
        output_layer = self.activated_nodes[len(self.activated_nodes) - 1]
        if len(target_layer) == len(output_layer):
            cost_layer = []
            for i in range(len(target_layer)):
                cost_layer.append((target_layer[i] - output_layer[i]) * (target_layer[i] - output_layer[i]))
            cost = 0
            for i in range(len(cost_layer)):
                cost += cost_layer[i]
            return cost
        else:
            print("COST SUM FUNCTION ERROR: target_layer and output_layer sizes do not match!")
            return None

    def cost_layer(self, target_layer):
        output_layer = self.activated_nodes[len(self.activated_nodes) - 1]
        if len(target_layer) == len(output_layer):
            cost_layer = []
            for i in range(len(target_layer)):
                cost_layer.append((1/2) * (target_layer[i] - output_layer[i]) * (target_layer[i] - output_layer[i]))
            return cost_layer
        else:
            print("COST LAYER FUNCTION ERROR: target_layer and output_layer sizes do not match!")
            return None

    def train(self, input_layer, target_layer):
        self.feed_forward(input_layer)

        # create new weights variable to later replace current weights
        new_weights = []
        for i in range(len(self.weights)):
            new_weights.append([])
            for j in range(len(self.weights[i])):
                new_weights[i].append([])

        # propagation error variables for back propagation
        propagation_error = []
        propagation_error_temp = []

        # back propagation
        for i in range(len(self.weights) - 1, -1, -1):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    if i == len(self.weights) - 1:
                        new_weights[i][j].append(self.weights[i][j][k] - learning_rate * (-1 * (target_layer[j] - self.activated_nodes[i + 1][j]) * NeuralNet.activation_function(self.inactive_nodes[i + 1][j], True) * self.activated_nodes[i][k]))
                        propagation_error.append(-1 * (target_layer[j] - self.activated_nodes[i + 1][j]) * NeuralNet.activation_function(self.inactive_nodes[i + 1][j], True) * self.weights[i][j][k])
                    else:
                        new_weights[i][j].append(self.weights[i][j][k] - learning_rate * (propagation_error[j] * NeuralNet.activation_function(self.inactive_nodes[i + 1][j], True) * self.activated_nodes[i][k]))
                        propagation_error_temp.append(propagation_error[j] * self.weights[i][j][k] * NeuralNet.activation_function(self.inactive_nodes[i + 1][j], True))
            if i != len(self.weights) - 1:
                propagation_error = propagation_error_temp
                propagation_error_temp = []

        self.weights = new_weights

    @staticmethod
    def find_best_structure(input_layer_array, target_layer_array, weight_scalar_range, hl_s_max_length, hl_s_max_height):
        # brute force function to find the best neural net architecture

        # hl_s -> hidden layer structure
        # input_layer_array & target_layer_array are 2d arrays of corresponding training data

        NeuralNet.trial_args_arr = []  # variable to store all possible combinations in our range
        for i in range(1, weight_scalar_range + 1):
            tried_nn_s = []  # array keep track of found
            print("i: " + str(i))
            NeuralNet.recursive_find_best_structure(input_layer_array, target_layer_array, i, hl_s_max_height, hl_s_max_length, tried_nn_s)

        # multiprocess all possible solutions in our range
        pool = mp.Pool(processor_use_count)
        results = []
        for i in range(len(NeuralNet.trial_args_arr)):
            results.append(pool.apply_async(NNStructureTester.run, args=(NeuralNet.trial_args_arr[i][0], NeuralNet.trial_args_arr[i][1], input_layer_array, target_layer_array)))
        for i in range(len(results) - 1, 0, -1):
            this_result = results[i].get()
            if this_result[1] > NNStructureTester.best_nn_s_accuracy:
                print("NEW BEST NN_S: " + str(this_result[0]) + "and weight scaling of: " + str(this_result[2]))
                print("\tWith an accuracy of: "+str(this_result[1]))
                NNStructureTester.best_nn_s = this_result[0]
                NNStructureTester.best_nn_s_accuracy = this_result[1]
                NNStructureTester.best_nn_s_weight_scalar = this_result[2]


    @staticmethod
    def recursive_find_best_structure(input_layer_array, target_layer_array, weight_scalar_range, hl_s_height, n, tried_nn_s, previous_hl_s=None):
        if previous_hl_s is None:
            previous_hl_s = []
        if n >= 1:
            for i in range(1, hl_s_height + 1):
                hl_s = deepcopy(previous_hl_s)
                hl_s.append(i)
                NeuralNet.recursive_find_best_structure(input_layer_array, target_layer_array, weight_scalar_range, hl_s_height, n - 1, tried_nn_s, hl_s)  # recursive call
        else:
            if previous_hl_s not in tried_nn_s:  # check to see if this hl_s has already been discovered
                tried_nn_s.append(previous_hl_s)  # add this hl_s to the array of already found hl_s's

                # create nn_s
                nn_s = [len(input_layer_array[0])]
                for i in range(len(previous_hl_s)):
                    nn_s.append(previous_hl_s[i])
                nn_s.append(len(target_layer_array[0]))

                # add this to the list of combinations to test
                this_trial_args = [nn_s, weight_scalar_range]
                NeuralNet.trial_args_arr.append(this_trial_args)

class NNStructureTester:
    best_nn_s = None
    best_nn_s_weight_scalar = 1
    best_nn_s_accuracy = 0

    max_attempts = 3

    @staticmethod
    def run(nn_s, weight_scalar, input_layer_array, target_layer_array):
        print("starting new trial! ["+str(weight_scalar)+"]["+str(nn_s)+"]")
        for attempt in range(NNStructureTester.max_attempts):  # allow each nn multiple attempts to get it right
            nn = NeuralNet(nn_s)
            nn.randomize_weights(weight_scalar)

            # train
            for i in range(len(input_layer_array)):
                nn.train(input_layer_array[i], target_layer_array[i])

            # test
            errors = []
            for i in range(len(input_layer_array)):
                output = nn.feed_forward(input_layer_array[i])
                for j in range(len(output)):
                    errors.append(1 - abs(output[j] - target_layer_array[i][j]))

            # evaluate
            sum = 0
            for i in range(len(errors)):
                sum += errors[i]
            this_accuracy = sum / len(errors)  # accuracy of this trial

            return nn_s, this_accuracy, weight_scalar


if __name__ == '__main__':
    nn_s = [3, 7, 5, 8, 8, 2, 1]
    train_amount = 10000

    input_layer_array = []
    target_layer_array = []

    # for j in range(train_amount):
    #     rand = random.randint(0, 7)
    #     if rand == 0:
    #         input_layer_array.append([0, 0, 0])
    #         target_layer_array.append([0])
    #     if rand == 1:
    #         input_layer_array.append([0, 0, 1])
    #         target_layer_array.append([.1428])
    #     if rand == 2:
    #         input_layer_array.append([0, 1, 0])
    #         target_layer_array.append([.2857])
    #     if rand == 3:
    #         input_layer_array.append([0, 1, 1])
    #         target_layer_array.append([.4286])
    #     if rand == 4:
    #         input_layer_array.append([1, 0, 0])
    #         target_layer_array.append([.5714])
    #     if rand == 5:
    #         input_layer_array.append([1, 0, 1])
    #         target_layer_array.append([.7143])
    #     if rand == 6:
    #         input_layer_array.append([1, 1, 0])
    #         target_layer_array.append([.8571])
    #     if rand == 7:
    #         input_layer_array.append([1, 1, 1])
    #         target_layer_array.append([1])

    # NeuralNet.optimize(input_layer_array, target_layer_array, 3, 5, 8)
    # print("Best Optimization was: " + str(NNStructureTester.best_nn_s) + " with a weight scaling of: " + str(NNStructureTester.best_nn_s_weight_scalar))
    # print("\tWith an accuracy of: "+ str(NNStructureTester.best_nn_s_accuracy))

    nn_1 = NeuralNet(nn_s)
    nn_1.randomize_weights(3)

    for j in range(train_amount):

        rand = random.randint(0, 7)
        if rand == 0:
            nn_1.train([0, 0, 0], [0])
        if rand == 1:
            nn_1.train([0, 0, 1], [.1428])
        if rand == 2:
            nn_1.train([0, 1, 0], [.2857])
        if rand == 3:
            nn_1.train([0, 1, 1], [.4286])
        if rand == 4:
            nn_1.train([1, 0, 0], [.5714])
        if rand == 5:
            nn_1.train([1, 0, 1], [.7143])
        if rand == 6:
            nn_1.train([1, 1, 0], [.8571])
        if rand == 7:
            nn_1.train([1, 1, 1], [1])

        if j % (train_amount * .01) == 0:
            print("Training "+str((j / train_amount) * 100)+"% complete")
            weight_list.append(nn_1.weights)

    print(str(nn_1.feed_forward([0, 0, 0])))
    print(str(nn_1.feed_forward([0, 0, 1])))
    print(str(nn_1.feed_forward([0, 1, 0])))
    print(str(nn_1.feed_forward([0, 1, 1])))
    print(str(nn_1.feed_forward([1, 0, 0])))
    print(str(nn_1.feed_forward([1, 0, 1])))
    print(str(nn_1.feed_forward([1, 1, 0])))
    print(str(nn_1.feed_forward([1, 1, 1])))