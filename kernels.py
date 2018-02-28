import numpy as np


class IdentityKernel:

    def value(self, x1, x2):
        return np.matmul(np.transpose(x1), x2)[0][0]

    def name(self):
        return "Identity"


class GaussianKernel:

    def __init__(self, sigma):
        self.sigma = sigma

    def value(self, x1, x2):
        return np.exp((-1 / 2.0) * ((np.linalg.norm(x1 - x2)) ** 2) / (self.sigma ** 2))

    def name(self):
        return "Gaussian with sigma =", self.sigma


class PolynomialKernel:

    def __init__(self, degree):
        self.degree = degree

    def value(self, x1, x2):
        return (np.matmul(np.transpose(x1), x2)[0][0] + 1)**self.degree

    def name(self):
        return "Polynomial with degree =", self.degree


if __name__ == '__main__':

    # Test Identity Kernel
    print('\n===Test Identity Kernel===')
    kernel = IdentityKernel()
    test_x1 = np.array([2, 3]).reshape((2, 1))
    test_x2 = np.array([5, 6]).reshape((2, 1))
    result = kernel.value(test_x1, test_x2)
    print('k(x1, x2) = ', result)

    # Test Gaussian Kernel
    print('\n===Test Gaussian Kernel===')
    kernel = GaussianKernel(sigma=3)
    test_x1 = np.array([2, 3]).reshape((2, 1))
    test_x2 = np.array([5, 6]).reshape((2, 1))
    result = kernel.value(test_x1, test_x2)
    print('k(x1, x2) = ', result)

    # Test Polynomial Kernel
    print('\n===Test Polynomial Kernel===')
    kernel = PolynomialKernel(degree=2)
    test_x1 = np.array([2, 3]).reshape((2, 1))
    test_x2 = np.array([5, 6]).reshape((2, 1))
    result = kernel.value(test_x1, test_x2)
    print('k(x1, x2) = ', result)
