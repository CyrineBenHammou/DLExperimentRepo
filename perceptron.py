import random

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.01):
        self.num_inputs = num_inputs # number of inputs
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)] # weights for each input
        self.bias = random.uniform(-1, 1) # bias of the perceptron
        self.learning_rate = learning_rate # step size for updating weights during training

# Takes a list of input values and calculates the weighted sum of inputs plus the bias
    def predict(self, inputs):
        sum = self.bias
        for i in range(self.num_inputs):
            sum += inputs[i] * self.weights[i]
        return 1 if sum > 0 else 0

# Trains the perceptron using the provided training data for a given number of epochs
    def train(self, training_data, epochs):
        for _ in range(epochs):
            for inputs, label in training_data:
                prediction = self.predict(inputs)
                error = label - prediction
                self.bias += self.learning_rate * error
                for i in range(self.num_inputs):
                    self.weights[i] += self.learning_rate * error * inputs[i]

# Example
if __name__ == "__main__":
    
    # Define training data: OR gate
    training_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1)
    ]
    
    # Create and train the perceptron
    perceptron = Perceptron(num_inputs=2)
    perceptron.train(training_data, epochs=1000)

    # Test the trained perceptron
    print("Testing OR gate:")
    for inputs, label in training_data:
        prediction = perceptron.predict(inputs)
        print(f"Inputs: {inputs}, Label: {label}, Predicted: {prediction}")
