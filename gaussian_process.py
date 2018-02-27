from data_loader import DataLoader
import numpy as np
import pandas as pd
import pprint
import kernels
import time as tm


class GaussianProcessRegressor:

    def __init__(self, input_vector_degree, kernel=None):
        self.M = input_vector_degree
        self.output_noise_mean = 0
        self.output_noise_variance = 1
        self.y_vector = None
        self.N = None
        self.gram_matrix = None
        self.modified_gram_matrix_inv = None
        self.mse_error = None
        self.training_time = None
        if kernel is None:
            self.kernel = kernels.IdentityKernel()
        else:
            self.kernel = kernel

    def compute_gram_matrix(self, dataset):

        full_train_set_attrs = []
        for train_set_attrs, train_set_labels in dataset:

            if len(train_set_attrs) != len(train_set_labels):
                raise ValueError('Count mismatch between attributes and labels')

            full_train_set_attrs.append(train_set_attrs)

        full_train_set_attrs = pd.concat(full_train_set_attrs, ignore_index=True)

        self.N = len(full_train_set_attrs)
        gram_matrix = np.empty(shape=(self.N, self.N), dtype=float)

        for i, row_i in full_train_set_attrs.iterrows():
            xi = row_i.values.reshape((self.M, 1))

            for j, row_j in full_train_set_attrs.iterrows():
                xj = row_j.values.reshape((self.M, 1))

                gram_matrix[i, j] = self.kernel.value(xi, xj)

        self.gram_matrix = gram_matrix
        return self.gram_matrix

    def compute_modified_gram_matrix_inv(self):

        K = self.gram_matrix
        self.modified_gram_matrix_inv = np.linalg.inv(np.add(K,
                                                             self.output_noise_variance**2 * np.identity(K.shape[0])))
        return self.modified_gram_matrix_inv

    def compute_kernel_with_dataset(self, dataset, new_x):

        k_with_X = []
        for train_set_attrs, train_set_labels in dataset:

            for i, row in train_set_attrs.iterrows():
                xi = row.values.reshape((self.M, 1))
                k_with_X.append(self.kernel.value(new_x, xi))

        k_with_X = np.array(k_with_X)
        k_with_X = np.reshape(k_with_X, (1, k_with_X.shape[0]))
        return k_with_X

    def posterior_mean_function(self, dataset, new_x):

        k_new_x_with_X = self.compute_kernel_with_dataset(dataset, new_x)

        return np.matmul(k_new_x_with_X, np.matmul(self.modified_gram_matrix_inv, self.y_vector))[0]

    def posterior_covariance_function(self, dataset, new_x):
        k_new_x_with_X = self.compute_kernel_with_dataset(dataset, new_x)
        k_X_with_new_x = np.transpose(k_new_x_with_X)

        return np.subtract(self.kernel.value(new_x, new_x),
                           np.matmul(k_new_x_with_X,
                                     np.matmul(self.modified_gram_matrix_inv, k_X_with_new_x)))[0][0]

    def learn(self, dataset, report_error=False):

        if report_error:
            start_time = tm.time()

        self.compute_gram_matrix(dataset)
        self.compute_modified_gram_matrix_inv()

        self.y_vector = []
        for train_set_attrs, train_set_labels in dataset:

            if len(train_set_attrs) != len(train_set_labels):
                raise ValueError('Count mismatch between attributes and labels')

            self.y_vector += train_set_labels.ix[:, 0].tolist()

        self.y_vector = np.array(self.y_vector, dtype=float)
        self.y_vector.reshape((self.y_vector.shape[0], 1))

        if report_error:
            self.mse_error = self.k_fold_cross_validation(dataset)
            print('Mean Square Error = %.3f ' % self.mse_error)
            end_time = tm.time()
            self.training_time = (int((end_time - start_time) * 100)) / 100
            print('Training time =', self.training_time, 'seconds')

    def predict_point(self, dataset, new_x):
        """
        uses the mean function of the posterior distribution for the prediction
        """
        return self.posterior_mean_function(dataset, new_x)

    def predict(self, train_dataset, test_attrs, true_values=None):

        if not true_values.empty:
            if len(test_attrs) != len(true_values):
                raise ValueError('count mismatch in attributes and labels')
            error = 0.0

        predicted_values = []
        for i, row in test_attrs.iterrows():
            xi = row.values.reshape((self.M, 1))
            predicted_value = self.predict_point(train_dataset, xi)
            predicted_values.append(predicted_value)
            if not true_values.empty:
                true_value = true_values.iat[i, 0]
                error += (true_value - predicted_value) ** 2

        E_MSE = None
        if true_values is not None:
            E_MSE = error / self.N

        predicted_values = pd.DataFrame(np.array(predicted_values))
        return predicted_values, E_MSE

    def k_fold_cross_validation(self, dataset, k=10):
        cv_test_model = GaussianProcessRegressor(input_vector_degree=self.M, kernel=self.kernel)
        avg_E_MSE = 0.0
        for i in range(k):
            test_attrs, test_labels = dataset.pop(0)
            cv_test_model.learn(dataset)
            E_MSE = cv_test_model.predict(dataset, test_attrs, true_values=test_labels)[1]
            dataset.append((test_attrs, test_labels))
            avg_E_MSE += E_MSE

        avg_E_MSE = avg_E_MSE / k
        return avg_E_MSE

    def summary(self):
        print('=====Model Summary=====')
        print('Input vector size =', self.M)
        print('Kernel =', self.kernel.name())
        print('\nGram Matrix of size', end=' ')
        print(self.gram_matrix.shape, ':')
        pprint.pprint(self.gram_matrix)
        if self.mse_error:
            print('Mean Square Error = %.3f ' % self.mse_error)


if __name__ == '__main__':

    # Test compute_gram_matrix
    print('\n===Test compute_gram_matrix===')
    kernel = kernels.PolynomialKernel(degree=2)
    model = GaussianProcessRegressor(input_vector_degree=2, kernel=kernel)
    train_attrs, train_labels = DataLoader.load_dataset(
        './regression-dataset/fData1.csv',
        './regression-dataset/fLabels1.csv'
    )
    result = model.compute_gram_matrix([(train_attrs, train_labels)])
    print('Gram Matrix shape=', result.shape)
    print('=====Gram Matrix====')
    pprint.pprint(result)

    # Test compute_modified_gram_matrix_inv
    print('\n===Test compute_modified_gram_matrix_inv===')
    kernel = kernels.PolynomialKernel(degree=2)
    model = GaussianProcessRegressor(input_vector_degree=2, kernel=kernel)
    train_attrs, train_labels = DataLoader.load_dataset(
        './regression-dataset/fData1.csv',
        './regression-dataset/fLabels1.csv'
    )
    model.compute_gram_matrix([(train_attrs, train_labels)])
    result = model.compute_modified_gram_matrix_inv()
    print('Matrix shape=', result.shape)
    print('=====Matrix====')
    pprint.pprint(result)

    # Test compute_kernel_with_dataset
    print('\n===Test compute_kernel_with_dataset===')
    model = GaussianProcessRegressor(input_vector_degree=2)
    train_attrs, train_labels = DataLoader.load_dataset(
        './regression-dataset/fData1.csv',
        './regression-dataset/fLabels1.csv'
    )
    test_x = np.array([2, 3]).reshape((2, 1))
    result = model.compute_kernel_with_dataset([(train_attrs, train_labels)], test_x)
    print('k(new_x, X) shape =', result.shape)
    print('=====k(new_x, X)====')
    pprint.pprint(result)

    # Test posterior_mean_function
    print('\n===Test posterior_mean_function===')
    model = GaussianProcessRegressor(input_vector_degree=2)
    train_attrs, train_labels = DataLoader.load_dataset(
        './regression-dataset/fData1.csv',
        './regression-dataset/fLabels1.csv'
    )
    test_x = np.array([7, 14]).reshape((2, 1))
    model.learn([(train_attrs, train_labels)])
    result = model.posterior_mean_function([(train_attrs, train_labels)], test_x)
    print('result =', result)

    # Test posterior_covariance_function
    print('\n===Test posterior_covariance_function===')
    model = GaussianProcessRegressor(input_vector_degree=2)
    train_attrs, train_labels = DataLoader.load_dataset(
        './regression-dataset/fData1.csv',
        './regression-dataset/fLabels1.csv'
    )
    test_x = np.array([7, 14]).reshape((2, 1))
    model.learn([(train_attrs, train_labels)])
    result = model.posterior_covariance_function([(train_attrs, train_labels)], test_x)
    print('result =', result)

    # Test predict_point
    print('\n===Test predict_point===')
    model = GaussianProcessRegressor(input_vector_degree=2)
    train_attrs, train_labels = DataLoader.load_dataset(
        './regression-dataset/fData1.csv',
        './regression-dataset/fLabels1.csv'
    )
    test_x = np.array([7, 14]).reshape((2, 1))
    model.learn([(train_attrs, train_labels)])
    result = model.predict_point([(train_attrs, train_labels)], test_x)
    print('result =', result)

    # Test GP with Identity Kernel
    print('\n===Test GP with Identity Kernel===')
    kernel = kernels.IdentityKernel()
    model = GaussianProcessRegressor(input_vector_degree=2, kernel=kernel)
    full_dataset = DataLoader.load_full_dataset('./regression-dataset')
    model.learn(full_dataset, report_error=True)
    model.summary()

    # Test GP with Gaussian Kernel
    print('\n===Test GP with Gaussian Kernel===')
    kernel = kernels.GaussianKernel(sigma=3)
    model = GaussianProcessRegressor(input_vector_degree=2, kernel=kernel)
    full_dataset = DataLoader.load_full_dataset('./regression-dataset')
    model.learn(full_dataset, report_error=True)
    model.summary()

    # Test GP with Polynomial Kernel
    print('\n===Test GP with Polynomial Kernel===')
    kernel = kernels.PolynomialKernel(degree=2)
    model = GaussianProcessRegressor(input_vector_degree=2, kernel=kernel)
    full_dataset = DataLoader.load_full_dataset('./regression-dataset')
    model.learn(full_dataset, report_error=True)
    model.summary()





