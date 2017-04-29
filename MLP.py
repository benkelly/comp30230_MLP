import numpy


class MLP:
    def __init__(self, ni, nh, no):
        self.no_of_inputs = ni                 # NI = (number of inputs)
        self.no_of_hidden_units = nh           # NH = (number of hidden units)
        self.no_of_outputs = no                # NO = (number of outputs)
        self.w1 = numpy.array                  # W1[][]  = (array containing the weights in the lower layer)
        self.w2 = numpy.array                  # W2[][] = (array containing the weights in the upper layer)
        self.dw1 = numpy.array                 # dW1[][](arr containing the weight *changes* to be applied onto W1 & W2)
        self.dw2 = numpy.array                 # dW2[][](arr containing the weight *changes* to be applied onto W1 & W2)
        self.z1 = numpy.array                  # Z1[] = (array containing the activations for the lower layer
        self.z2 = numpy.array                  # Z2[] = (array containing the activations for the upper layer
        self.h = numpy.array                   # H[] = (array where the values of the hidden neurons are stored)
        self.o = numpy.array                   # O[] = (array where the outputs are stored)

    # sigmoid for range 0..1
    def activation_sigmoid(self, sig_input, der=False):
        if der:
            return numpy.exp(-sig_input) / (1 + numpy.exp(-sig_input)) ** 2
        else:
            return 1 / (1 + numpy.exp(-sig_input))

    # tanh for range -1..1
    def activation_tanh(self, tanh_input, der=False):
        if der:
            return 1 - (numpy.power(self.activation_tanh(tanh_input), 2))
        else:
            return (2 / (1 + numpy.exp(tanh_input * -2))) - 1

    # initialises weights to small random values, and fills deltas with
    # 0s to the same size matrix as the corresponding weight matrix
    def randomise(self):
        self.w1 = numpy.array((numpy.random.uniform(0.0, 1, (self.no_of_inputs, self.no_of_hidden_units))).tolist())
        self.dw1 = numpy.dot(self.w1, 0)
        self.w2 = numpy.array((numpy.random.uniform(0.0, 1, (self.no_of_hidden_units, self.no_of_outputs))).tolist())
        self.dw2 = numpy.dot(self.w2, 0)

    # forward pass, computes activations in z1 and z2, then fills h and o with the activated
    # values either sigmoid or tanh depending on whether it is the xor or sine problem
    def forward(self, input_vectors, sin=False):
        self.z1 = numpy.dot(input_vectors, self.w1)
        if sin:
            self.h = self.activation_tanh(self.z1)
        else:
            self.h = self.activation_sigmoid(self.z1)

        self.z2 = numpy.dot(self.h, self.w2)
        if sin:
            self.o = self.activation_tanh(self.z2)
        else:
            self.o = self.activation_sigmoid(self.z2)

    def backwards(self, input_vectors, target, sin=False):

        output_error = numpy.subtract(target, self.o)

        if sin:
            activation_lower_layer = self.activation_tanh(self.z1, True)
            activation_upper_layer = self.activation_tanh(self.z2, True)
        else:
            activation_lower_layer = self.activation_sigmoid(self.z1, True)
            activation_upper_layer = self.activation_sigmoid(self.z2, True)

        dw2_a = numpy.multiply(output_error, activation_upper_layer)
        self.dw2 = numpy.dot(self.h.T, dw2_a)

        dw1_a = numpy.multiply(numpy.dot(dw2_a, self.w2.T), activation_lower_layer)
        self.dw1 = numpy.dot(input_vectors.T, dw1_a)

        return numpy.mean(numpy.abs(output_error))

    # updates weights with the regard to learning rate after the deltas computed in backwards()
    def update_weights(self, learning_rate):
        self.w1 = numpy.add(self.w1, learning_rate * self.dw1)
        self.w2 = numpy.add(self.w2, learning_rate * self.dw2)
        self.dw1 = numpy.array
        self.dw2 = numpy.array

    def info_to_string(self):
        print('(NI) Number of Inputs: \n' + str(self.no_of_inputs))
        print('(NH) Number of Hidden Units: \n' + str(self.no_of_hidden_units))
        print('(NO) Number of Outputs: \n' + str(self.no_of_outputs))
        print('w1: \n' + str(self.w1))
        print('w2: \n' + str(self.w2))
        print('dw1: \n' + str(self.dw1))
        print('dw2: \n' + str(self.dw2))
        print('z1: \n' + str(self.z1))
        print('z2: \n' + str(self.z2))
        print('h: \n' + str(self.h))
        print('o: \n' + str(self.o))
        print('\n')