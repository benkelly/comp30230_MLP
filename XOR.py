import numpy
from MLP import MLP

# for testing,
# numpy.random.seed(1)


# XOR training data
# desired output:
xor_inputs = numpy.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_desired_output = numpy.array([[0], [1], [1], [0]])


MAX_EPOCHS = 100000
LEARNING_RATE = 1
# set up mlp and set initial weights
INPUTS = 2
HIDDEN = 2
OUTPUTS = 1
mlp = MLP(INPUTS, HIDDEN, OUTPUTS)
mlp.randomise()

# saves output to file
with open('test_output/xor_output/xor_output_size_(' + str(INPUTS) + ', ' + str(HIDDEN) + ', ' + str(OUTPUTS)
                  + ')_learning_rate_' + str(LEARNING_RATE) + '_epochs_' + str(MAX_EPOCHS) + '.txt', 'w') as f:

    print('\nPre-Training Testing:\n')
    f.write('\nPre-Training Testing:\n')
    for i in range(len(xor_inputs)):
        mlp.forward(xor_inputs[i])
        print('Target:\t' + str(xor_desired_output[i]) + '\t\tOutput:\t' + str(mlp.o) + '\n')
        f.write('Target:\t' + str(xor_desired_output[i]) + '\t\tOutput:\t' + str(mlp.o) + '\n')

    f.write('MLP Size\t(' + str(INPUTS) + ', ' + str(HIDDEN) + ', ' + str(OUTPUTS) + ')\n\n')
    f.write('Epochs:\t\t' + str(MAX_EPOCHS) + '\n')
    f.write('Learning Rate:\t' + str(LEARNING_RATE) + '\n\n')

    print('Training:\n')
    f.write('Training:\n')
    for i in range(0, MAX_EPOCHS):
        error = 0
        mlp.forward(xor_inputs)
        error = mlp.backwards(xor_inputs, xor_desired_output)
        mlp.update_weights(LEARNING_RATE)

        if (i + 1) % (MAX_EPOCHS / 10) == 0:
            length = len(str(i))
            print('Epoch:\t' + str(i + 1) + '\tError:\t' + str(error))
            f.write('Epoch:\t' + str(i + 1) + '\tError:\t' + str(error) + '\n')

    print('\nTesting:\n')
    f.write('\nTesting:\n')
    for i in range(len(xor_inputs)):
        mlp.forward(xor_inputs[i])
        mlp.forward(xor_inputs[i])
        print('Target:\t' + str(xor_desired_output[i]) + '\t\tOutput:\t' + str(mlp.o) + '\n')
        f.write('Target:\t' + str(xor_desired_output[i]) + '\t\tOutput:\t' + str(mlp.o) + '\n')
