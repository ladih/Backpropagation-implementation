# Testing the network

from network import *
import random

# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[0], [1], [1], [0]])

X = np.array([[0, 0], [0.5, 0], [1, 0], [0, 0.5], [0, 1], [0.25, 0.25], [0.5, 0.5], [1, 1]])    # input
y = np.array([[1], [1], [1], [1], [1], [0], [0], [0]])     # labels

# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])    # input
# y = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0], [0, 0, 1]])

# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.5, 0.5], [0.5, 1], [0, 0.5], [0.5, 0], [1, 0.5]])    # input
# y = np.array([[1], [1], [1], [1], [0], [1], [1], [1], [1]])

nn = NeuralNetwork(input_size=len(X[0]), output_size=len(y[0]), hidden_sizes=[3, 100, 100, 3])

nn.train(X, y, learning_rate=0.1, epochs=200000, batch_size=len(X), error_threshold=0.0001)


print("\nSome predictions:")
print(nn.predict(X[0]))
print(nn.predict(X[1]))
print(nn.predict(X[2]))
print(nn.predict(X[3]))

nn.plot_error_curve()

def plot_2D(n):
    """If input is 2D and output is 1D, this function shows how n
        random points in the plane are classified (assuming labels are 0 or 1)"""
    xmin, xmax = -0.5, 1.5
    ymin, ymax = -0.5, 1.5
    for i in range(n):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        res = nn.predict(np.array([[x, y]]))
        if res > 0.5:
            plt.scatter(x, y, color='black')
        else:
            plt.scatter(x, y, color='red')
    plt.show()


if len(X[0]) == 2 and len(y[0]) == 1:
    plot_2D(500)
