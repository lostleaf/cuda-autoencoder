import numpy as np
import matplotlib.pyplot as plt
import time
import cPickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def KL_divergence(x, y):
    return x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))

def load_MNIST_images(filename):
    with open(filename, "r") as f:
        magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        num_images = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        num_rows = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
        num_cols = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

        images = np.fromfile(f, dtype=np.ubyte)
        images = images.reshape((num_images, num_rows * num_cols)).transpose()
        images = images.astype(np.float64) / 255

        f.close()

        return images

class AutoEncoder:
    def __init__(self, visible_size, hidden_size):
        self.visible_size = visible_size
        self.hidden_size = hidden_size

        r = np.sqrt(6) / np.sqrt(hidden_size + visible_size + 1)
        self.W1 = np.random.random((hidden_size, visible_size)) * 2 * r - r
        self.W2 = np.random.random((visible_size, hidden_size)) * 2 * r - r

        self.b1 = np.zeros((hidden_size, 1), dtype=np.float64)
        self.b2 = np.zeros((visible_size, 1), dtype=np.float64)
        np.savetxt("W1.txt", self.W1)
        np.savetxt("W2.txt", self.W2)
        np.savetxt("b1.txt", self.b1)
        np.savetxt("b2.txt", self.b2)

    def sparse_autoencoder_cost(self, lambda_, rho, beta, data):
        # Number of training examples
        m = data.shape[1]

        # Forward propagation
        z2 = np.dot(self.W1, data) + self.b1
        a2 = sigmoid(z2)
        z3 = np.dot(self.W2, a2) + self.b2
        a3 = sigmoid(z3)

        # Sparsity
        rho_hat = np.sum(a2, axis=1, keepdims=True) / m
        # rho = np.tile(sparsity_param, hidden_size)
        # print rho_hat

        # Cost function
        cost = np.sum((a3 - data) ** 2) / (2 * m) +  (lambda_ / 2) * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2)) +  beta * np.sum(KL_divergence(rho, rho_hat))
        # cost = np.sum((a3 - data) ** 2) / (2 * m) 

        # Backprop
        delta_sparse = - rho / rho_hat + (1 - rho) / (1 - rho_hat)

        delta3 = -(data - a3) * sigmoid_prime(z3)
        delta2 = (np.dot(self.W2.T, delta3) + beta * delta_sparse) * sigmoid_prime(z2)
        print delta2
        grad_W1 = np.dot(delta2, data.T) / m + lambda_ * self.W1
        grad_W2 = np.dot(delta3, a2.T) / m + lambda_ * self.W2
        grad_b1 = np.sum(delta2, axis=1, keepdims=True) / m
        grad_b2 = np.sum(delta3, axis=1, keepdims=True) / m 
        return cost, grad_W1, grad_b1, grad_W2, grad_b2
    
    def reconstruct(self, data):
        z2 = np.dot(self.W1, data) + self.b1
        a2 = sigmoid(z2)
        z3 = np.dot(self.W2, a2) + self.b2
        a3 = sigmoid(z3)
        return a3

def main():
    np.random.seed(42)
    visible_size = 28 * 28
    hidden_size = 196

    sparsity_param = 0.1
    lambda_ = 3e-3
    beta = 3

    images = load_MNIST_images('train-images-idx3-ubyte')
    patches = images[:, 0:1000]

    ae = AutoEncoder(visible_size, hidden_size)

    rate = 0.5
    MAX_IT = 1
    t1 = time.time()
    for i in xrange(MAX_IT):
        cost, grad_W1, grad_b1, grad_W2, grad_b2 = ae.sparse_autoencoder_cost(lambda_, sparsity_param, beta, patches)
        ae.W1 -= rate * grad_W1
        ae.W2 -= rate * grad_W2
        ae.b1 -= rate * grad_b1
        ae.b2 -= rate * grad_b2
        if (i) % 100 == 0:
            t2 = time.time()
            rate *= 0.95
            print i + 1, cost, (t2 - t1) * 1000000
            t1 = time.time()
    # with open("ae.pkl", "wb") as fout:
    #     cPickle.dump(ae, fout, 2)

if __name__ == "__main__":
    main()
