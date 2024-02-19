import random
import math

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden, num_outputs, learning_rate=0.1):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        
        # Initialize weights for input layer to hidden layer
        self.weights_ih = [[random.uniform(-1, 1) for _ in range(num_hidden)] for _ in range(num_inputs)]
        # Initialize weights for hidden layer to output layer
        self.weights_ho = [[random.uniform(-1, 1) for _ in range(num_outputs)] for _ in range(num_hidden)]
        
        # Initialize biases for hidden layer
        self.bias_h = [random.uniform(-1, 1) for _ in range(num_hidden)]
        # Initialize biases for output layer
        self.bias_o = [random.uniform(-1, 1) for _ in range(num_outputs)]
        
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, inputs):
        # Calculate outputs of hidden layer
        hidden_outputs = [0] * self.num_hidden
        for i in range(self.num_hidden):
            sum = self.bias_h[i]
            for j in range(self.num_inputs):
                sum += inputs[j] * self.weights_ih[j][i]
            hidden_outputs[i] = self.sigmoid(sum)
        
        # Calculate outputs of output layer
        outputs = [0] * self.num_outputs
        for i in range(self.num_outputs):
            sum = self.bias_o[i]
            for j in range(self.num_hidden):
                sum += hidden_outputs[j] * self.weights_ho[j][i]
            outputs[i] = self.sigmoid(sum)
        
        return outputs

    def train(self, training_data, epochs):
        for _ in range(epochs):
            for inputs, targets in training_data:
                # Forward pass
                hidden_outputs = [0] * self.num_hidden
                for i in range(self.num_hidden):
                    sum = self.bias_h[i]
                    for j in range(self.num_inputs):
                        sum += inputs[j] * self.weights_ih[j][i]
                    hidden_outputs[i] = self.sigmoid(sum)

                outputs = [0] * self.num_outputs
                for i in range(self.num_outputs):
                    sum = self.bias_o[i]
                    for j in range(self.num_hidden):
                        sum += hidden_outputs[j] * self.weights_ho[j][i]
                    outputs[i] = self.sigmoid(sum)

                # Backpropagation
                output_errors = [targets[i] - outputs[i] for i in range(self.num_outputs)]
                output_gradients = [self.sigmoid_derivative(outputs[i]) * output_errors[i] for i in range(self.num_outputs)]

                hidden_errors = [0] * self.num_hidden
                for i in range(self.num_hidden):
                    error = 0
                    for j in range(self.num_outputs):
                        error += output_gradients[j] * self.weights_ho[i][j]
                    hidden_errors[i] = error
                
                hidden_gradients = [self.sigmoid_derivative(hidden_outputs[i]) * hidden_errors[i] for i in range(self.num_hidden)]

                # Update weights and biases
                for i in range(self.num_hidden):
                    for j in range(self.num_outputs):
                        self.weights_ho[i][j] += self.learning_rate * output_gradients[j] * hidden_outputs[i]
                    self.bias_o[j] += self.learning_rate * output_gradients[j]

                for i in range(self.num_inputs):
                    for j in range(self.num_hidden):
                        self.weights_ih[i][j] += self.learning_rate * hidden_gradients[j] * inputs[i]
                    self.bias_h[j] += self.learning_rate * hidden_gradients[j]

# Example 
if __name__ == "__main__":
    
    # Define training data: XOR gate
    training_data = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]),
        ([1, 1], [0])
    ]
    
    # Create and train the neural network
    nn = NeuralNetwork(num_inputs=2, num_hidden=2, num_outputs=1)
    nn.train(training_data, epochs=10000)

    # Test the trained neural network
    print("Testing XOR gate:")
    for inputs, targets in training_data:
        prediction = nn.feedforward(inputs)[0]
        print(f"Inputs: {inputs}, Targets: {targets}, Predicted: {prediction}")
