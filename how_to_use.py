# Example usage

from network import *  # import network

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # training data
y = np.array([[0], [1], [1], [0]])              # labels

# Create an instance of the NeuralNetwork class
nn = NeuralNetwork(input_size=len(X[0]), output_size=len(y[0]), hidden_sizes=[5, 10, 5])

# Train the instance
nn.train(X, y, learning_rate=0.1, epochs=200000, batch_size=len(X), error_threshold=0.001)

# Predict unseen data
pred = nn.predict([0.8, 0.8])
