import numpy as np

class NeuronNetwork:
    def __init__(self, input_size=50, hidden_layers=[3, 10], output_size=3):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        
        # Input to hidden layers network
        self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.biases.append(np.zeros((1, hidden_layers[0])))
        
        # Hidden layers network
        for i in range(len(hidden_layers) - 1):
            self.weights.append(0.01 * np.random.randn(hidden_layers[i], hidden_layers[i + 1]))
            self.biases.append(np.zeros((1, hidden_layers[i + 1])))
        
        # Hidden layers network to output
        self.weights.append(0.01 * np.random.randn(hidden_layers[-1], output_size))
        self.biases.append(np.zeros((1, output_size)))
        print("Network initialized")

    def forward(self, inputs):
        layers = [inputs]
        print("Input Layer:\n", inputs)  # Print the input layer
        
        for i in range(len(self.weights)):
            # Compute the output of the current layer
            current_layer = np.dot(layers[-1], self.weights[i]) + self.biases[i]
            
            # Print the current layer
            print(f"Layer {i + 1} Output:\n", current_layer)
            
            # Append the current layer to the layers list
            layers.append(current_layer)
        
        return layers[-1]

# Example usage:
# Create a network
network = NeuronNetwork(input_size=100, hidden_layers=[3, 10], output_size=10)

# Define an input (you can adjust this based on your input size)
input_data = np.random.randn(1, 100)

# Perform forward propagation and print layers
output = network.forward(input_data)
