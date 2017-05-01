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

    # sets weights to random values and fills deltas with zeros
    # to the same size as weight matrix
    def randomise(self):
        self.w1 = numpy.array((numpy.random.uniform(0.0, 1, (self.no_of_inputs, self.no_of_hidden_units))).tolist())
        self.dw1 = numpy.dot(self.w1, 0)
        self.w2 = numpy.array((numpy.random.uniform(0.0, 1, (self.no_of_hidden_units, self.no_of_outputs))).tolist())
        self.dw2 = numpy.dot(self.w2, 0)

    # computes activations of z1,z2. fills h and o with the activated values
    # if SIN ->hyperbolic_tangent
    # if XOR ->logistic_sigmoid
    def forward(self, input_vectors, sin):
        self.z1 = numpy.dot(input_vectors, self.w1)
        if sin:
            self.h = self.hyperbolic_tangent(self.z1)
        else:
            self.h = self.logistic_sigmoid(self.z1)

        self.z2 = numpy.dot(self.h, self.w2)
        if sin:
            self.o = self.hyperbolic_tangent(self.z2)
        else:
            self.o = self.logistic_sigmoid(self.z2)

    # computes error. computes activation derivatives based on XOR or SIN.
    # if SIN ->hyperbolic_tangent
    # if XOR ->logistic_sigmoid
    # then multiplies the derivatives by the error
    # dot product of those values with h & o values produce the deltas
    def backwards(self, input_vectors, target, sin):
        output_error = numpy.subtract(target, self.o)
        if sin:
            activation_lower_layer = self.hyperbolic_tangent(self.z1, True)
            activation_upper_layer = self.hyperbolic_tangent(self.z2, True)
        else:
            activation_lower_layer = self.logistic_sigmoid(self.z1, True)
            activation_upper_layer = self.logistic_sigmoid(self.z2, True)

        dw2_a = numpy.multiply(output_error, activation_upper_layer)
        self.dw2 = numpy.dot(self.h.T, dw2_a)

        dw1_a = numpy.multiply(numpy.dot(dw2_a, self.w2.T), activation_lower_layer)
        self.dw1 = numpy.dot(input_vectors.T, dw1_a)

        return numpy.mean(numpy.abs(output_error))

    # updates weights of learning rate with new deltas computed in backwards()
    def update_weights(self, learning_rate):
        self.w1 = numpy.add(self.w1, learning_rate * self.dw1)
        self.w2 = numpy.add(self.w2, learning_rate * self.dw2)
        self.dw1 = numpy.array
        self.dw2 = numpy.array

    # Sigmoid of [0..1]
    def logistic_sigmoid(self, sig_input, backward=False):
        if backward:
            return numpy.exp(-sig_input) / (1 + numpy.exp(-sig_input)) ** 2
        else:
            return 1 / (1 + numpy.exp(-sig_input))

    # Tanh of [-1..1]
    def hyperbolic_tangent(self, tanh_input, backward=False):
        if backward:
            return 1 - (numpy.power(self.hyperbolic_tangent(tanh_input), 2))
        else:
            return (2 / (1 + numpy.exp(tanh_input * -2))) - 1

    # to string
    def __str__(self):
        str_temp = ("\n(NI) Number of Inputs: \t" + str(self.no_of_inputs))
        str_temp += ("\n(NH) Number of Hidden Units: \t" + str(self.no_of_hidden_units))
        str_temp += ("\n(NO) Number of Outputs: \t" + str(self.no_of_outputs))
        str_temp += ("\nw1: \t" + str(self.w1))
        str_temp += ("\nw2: \t" + str(self.w2))
        str_temp += ("\ndw1: \t" + str(self.dw1))
        str_temp += ("\ndw2: \t" + str(self.dw2))
        str_temp += ("\nz1: \t" + str(self.z1))
        str_temp += ("\nz2: \t" + str(self.z2))
        str_temp += ("\nh: \t" + str(self.h))
        str_temp += ("\no: \t" + str(self.o))
        str_temp += '\n'
        return str_temp
