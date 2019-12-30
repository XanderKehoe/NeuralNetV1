import random
import math

weight_list = [] # for debugging only
learning_rate = .3

class NeuralNet:
    # nn_structure is an array where each value specifies the size of each node (input, hidden, and output) layer.
    def __init__(self, nn_structure):
        self.actived_nodes = [0 for col in range(len(nn_structure))]
        for i in range(len(self.actived_nodes)):
            self.actived_nodes[i] = [0 for row in range(nn_structure[i])]
        self.inactive_nodes = self.actived_nodes

        self.weights = []
        for i in range(len(self.actived_nodes) - 1):
            self.weights.append([])
            for j in range(len(self.actived_nodes[i + 1])):
                self.weights[i].append([])
                for k in range(len(self.actived_nodes[i])):
                    self.weights[i][j].append(0)

    def randomize_weights(self, scalar):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] = scalar * ((2 * random.random()) - 1)  # generated random # between -1 and 1 times the scalar

    def print_nodes(self):
        for i in self.actived_nodes:
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
        if len(input_layer) == len(self.actived_nodes[0]):
            self.actived_nodes[0] = input_layer
            for i in range(len(self.actived_nodes) - 1):
                for j in range(len(self.actived_nodes[i + 1])):
                    #  tally up sum of weights * inputs to feed forward to next node layer
                    sum = 0;
                    for k in range(len(self.actived_nodes[i])):
                        sum += self.actived_nodes[i][k] * self.weights[i][j][k]
                    self.inactive_nodes[i + 1][j] = sum
                    self.actived_nodes[i+1][j] = self.sigmoid(self.inactive_nodes[i+1][j], False)
            return self.actived_nodes[len(self.actived_nodes) - 1]
        else:
            print("FEED FORWARD ERROR: input and input layer sizes do not match!")

    @staticmethod
    def sigmoid(x, derivative):
        if not derivative:
            return 1/(1+math.exp(-x))
        else:
            return math.exp(-x)/(math.pow(1+math.exp(-x), 2))

    def cost_sum(self, target_layer):
        output_layer = self.actived_nodes[len(self.actived_nodes) - 1]
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
        output_layer = self.actived_nodes[len(self.actived_nodes) - 1]
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
        cost_layer = self.cost_layer(target_layer)
        new_weights = []
        for i in range(len(self.weights)):
            new_weights.append([])
            for j in range(len(self.weights[i])):
                new_weights[i].append([])

        propagation_error = []
        propagation_error_temp = []

        for i in range(len(self.weights) - 1, -1, -1):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    if i == len(self.weights) - 1:
                        new_weights[i][j].append(self.weights[i][j][k] - learning_rate * (-1 * (target_layer[j] - self.actived_nodes[i+1][j]) * NeuralNet.sigmoid(self.inactive_nodes[i+1][j], True) * self.actived_nodes[i][k]))
                        propagation_error.append(-1 * (target_layer[j] - self.actived_nodes[i+1][j]) * NeuralNet.sigmoid(self.inactive_nodes[i+1][j], True) * self.weights[i][j][k])
                    else:
                        new_weights[i][j].append(self.weights[i][j][k] - learning_rate * (propagation_error[j] * NeuralNet.sigmoid(self.inactive_nodes[i+1][j], True) * self.actived_nodes[i][k]))
                        propagation_error_temp.append(propagation_error[j] * self.weights[i][j][k] * NeuralNet.sigmoid(self.inactive_nodes[i+1][j], True))
                        # new_weights[i][j].append(self.weights[i][j][k])  # no change in weights, DEBUGGING ONLY
            if i != len(self.weights) - 1:
                propagation_error = propagation_error_temp
                propagation_error_temp = []

        self.weights = new_weights


nn_s = [1, 7, 1]
train_amount = 10000

nn_1 = NeuralNet(nn_s)
nn_1.randomize_weights(10)

for j in range(train_amount):
    rand = random.random()
    nn_1.train([rand], [1-rand])

    if j % (train_amount * .01) == 0:
        print("Training "+str((j / train_amount) * 100)+"% complete")
        weight_list.append(nn_1.weights)

print(str(nn_1.feed_forward([0])))
print(str(nn_1.feed_forward([0.5])))
print(str(nn_1.feed_forward([1])))
