import numpy
from MLP import MLP

# for testing,
# numpy.random.seed(1)

# Generate the vectors and the desired output
sin_inputs = []
sin_desired_output = []

# 50 vectors containing 4 components each. The value of each component should be a random number between -1 and 1.
# These will be your input vectors. The corresponding output for each vector should be the sin() of the sum
# of the components. That is, for inputs:
# [x1 x2 x3 x4]
# the (single component) output should be:
# sin(x1+x2+x3+x4)
for i in range(0, 50):
    vector = list(numpy.random.uniform(-1.0, 1.0, 4))
    vector = [float(vector[0]), float(vector[1]), float(vector[2]), float(vector[3])]
    sin_inputs.append(vector)

sin_inputs = numpy.array(sin_inputs)

for i in range(0, 50):
    summed_vector_comps = numpy.sum(sin_inputs)
    sin_desired_output.append([numpy.sin(numpy.sum(sin_inputs[i]))])

sin_desired_output = numpy.array(sin_desired_output)


MAX_EPOCHS = 1000000
LEARNING_RATE = 0.005
# LEARNING_RATE = 0.05

# set up mlp and set initial weights
INPUTS = 4
HIDDEN = 5
OUTPUTS = 1
mlp = MLP(INPUTS, HIDDEN, OUTPUTS)
mlp.randomise()

# saves output to file
with open('test_output/sin_output/sin_output_size_(' + str(INPUTS) + ', ' + str(HIDDEN) + ', ' + str(OUTPUTS) + ')_learning_rate_'
                  + str(LEARNING_RATE) + '_epochs_' + str(MAX_EPOCHS) + '.txt', 'w') as f:

    print('\nPre-Training Testing:\n')
    f.write('\nPre-Training Testing:\n')
    for i in range(len(sin_inputs) - 10, len(sin_inputs)):
        mlp.forward(sin_inputs[i], True)
        print('Target:\t' + str(sin_desired_output[i]) + '\t\tOutput:\t' + str(mlp.o) + '\n')
        f.write('Target:\t' + str(sin_desired_output[i]) + '\t\tOutput:\t' + str(mlp.o) + '\n')

    f.write('MLP Size\t\t\t(' + str(INPUTS) + ', ' + str(HIDDEN) + ', ' + str(OUTPUTS) + ')\n')
    f.write('Epochs:\t\t\t\t' + str(MAX_EPOCHS) + '\n')
    f.write('Learning Rate:\t\t' + str(LEARNING_RATE) + '\n\n')

    print('Training:\n')
    f.write('Training:\n')
    for i in range(0, MAX_EPOCHS):
        error = 0
        mlp.forward(sin_inputs[:len(sin_inputs) - 10], True)

        error = mlp.backwards(sin_inputs[:(len(sin_inputs) - 10)], sin_desired_output[:len(sin_inputs) - 10],
                              True)
        mlp.update_weights(LEARNING_RATE)

        if (i + 1) % (MAX_EPOCHS / 10) == 0:
            print('Epoch:\t' + str(i + 1) + '\tError:\t' + str(error))
            f.write('Epoch:\t' + str(i + 1) + '\tError:\t' + str(error) + '\n')

    print('\nTesting:\n')
    for i in range(len(sin_inputs) - 10, len(sin_inputs)):
        mlp.forward(sin_inputs[i], True)
        print('Target:\t' + str(sin_desired_output[i]) + '\t\tOutput:\t' + str(mlp.o) + '\n')
        f.write('Target:\t' + str(sin_desired_output[i]) + '\t\tOutput:\t' + str(mlp.o) + '\n')
