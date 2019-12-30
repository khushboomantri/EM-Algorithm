
import argparse
import pandas as pd
import numpy as np
import scipy.stats as sp
from sklearn import metrics


def pass_parameters():
    '''
    @topic: Pass the arguments on terminal.
    '''
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('dataset', help="Select a dataset(txt file) as input.")
    parser.add_argument('k_value', help="Set the number of clusters.", type=int)
    args = parser.parse_args()
    dataset_name = args.dataset # dataset_name = 'Iris.csv'
    k_val = args.k_value
    return dataset_name, k_val


def input_dataset(dataset_name):
    '''
    @topic: Input and preprocess the dataset.
    @parameters: dataset_name: the name of dataset.
    '''
    data = pd.read_csv(dataset_name, header = 0)
    data = data.reset_index()
    # Replace the string label to 0,1,2
    replace_map = {'Species': {'Iris-virginica': 0, 'Iris-versicolor': 1,'Iris-setosa':2}}
    data.replace(replace_map, inplace=True)
    # ground truth label Y
    Y = data[['Species']]
    # Select 4 columns as the features
    col = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    X = data[col]
    X = np.array(X)
    return X, Y


class EM_Algorithm(object):
    def __init__(self, X, k=3, init_rd=True):
        '''
        @topic: Declare the initial parameters.
        @parameters:
            1. X: Dataset (without label).
            2. k: k value refers to the number of initial centroids.
            3. init: Whether randomly initialize the mean or not. (Default: random)
        '''
        X = np.asarray(X)
        self.m, self.n = X.shape # m: rows, n: columns
        self.data = X.copy()
        self.k = k # number of clusters given
        # Initialize prior and weight
        self.phi = np.ones(self.k)/self.k
        self.W = np.asmatrix(np.empty((self.m, self.k), dtype=float)) # m*k
        # Initialize mean and full covariance matrix
        if not init_rd:
            # Use np.asmatrix to avoid psd error
            # ValueError: the input matrix must be positive semidefinite
            X_k = self.k_fold(X, self.k)
            mean_ar = [np.mean(X_k[i], axis=0) for i in range(len(X_k))]
            cov_ar = [np.asmatrix(np.cov(X_k[i].T)) for i in range(len(X_k))]
            self.mean_arr = np.asmatrix(mean_ar)
            self.sigma_arr = np.array(cov_ar)
        else:    
            self.mean_arr = np.asmatrix(np.random.random((self.k, self.n))+np.mean(self.data)) # k*n
            self.sigma_arr = np.array([np.asmatrix(np.identity(self.n)) for i in range(self.k)]) # k*n*n


    def k_fold(self, X, k):
        '''
        @topic: Cut the dataset into k folds.
        @parameters:
            1. X: Dataset.
            2. k: The number of folds.
        '''
        # Use np.ceil to guarantee k fold.
        return [X[i:i + int(np.ceil(len(X)/k))] for i in range(0, len(X), int(np.ceil(len(X)/k)))]


    def em_clustering(self, tol=1e-8):
        '''
        @topic: Implement EM algorithm for clustering.
        @parameters: tol: tolerance used as your stopping condition.
        '''
        def logllh():
            # Compute the loglikelihood
            logl = 0
            for i in range(self.m):
                tmp = 0
                for j in range(self.k):
                    tmp += sp.multivariate_normal.pdf(self.data[i, :], self.mean_arr[j, :].A1, self.sigma_arr[j, :]) * self.phi[j]
                logl += np.log(tmp)
            return logl
        num_iters = 0 # Number of iterations
        logl = 1
        previous_logl = 0
        # Algorithm will run until the log-likelihood converges.
        while(logl - previous_logl > tol):
            previous_logl = logllh()
            # E-step: Compute the posterior and the weights.
            for i in range(self.m):
                den = 0
                for j in range(self.k):
                    likelihood = sp.multivariate_normal.pdf(self.data[i, :], self.mean_arr[j].A1, self.sigma_arr[j])
                    posterior_j = likelihood * self.phi[j]
                    den += posterior_j
                    self.W[i, j] = posterior_j
                self.W[i, :] /= den
            # M-step: Update the mean and cov.
            for j in range(self.k):
                const = self.W[:, j].sum()
                self.phi[j] = 1/self.m * const
                _mu_j = np.zeros(self.n)
                _sigma_j = np.zeros((self.n, self.n))
                for i in range(self.m):
                    _mu_j += (self.data[i, :] * self.W[i, j])
                    _sigma_j += self.W[i, j] * ((self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))
                self.mean_arr[j] = _mu_j / const
                self.sigma_arr[j] = _sigma_j / const
            num_iters += 1
            logl = logllh()
        # Label prediction
        predicted_label = []
        for i in range(len(self.W)):
            predicted_label.append(np.argmax(self.W[i], axis=1)[0,0])
        #print("predicted_label: ", predicted_label)
        return self.mean_arr, self.sigma_arr, predicted_label, num_iters


    def cluster_membership(self, y_pred):
        '''
        @topic: Assign the clusters and compute their size.
        @parameters: y_pred: predicted label. 
        '''
        cluster = [[] for i in range(self.k)]
        size = []
        class_ = list(set(y_pred))
        for i in range(len(y_pred)):
            for j in range(len(class_)):
                if y_pred[i] == class_[j]:
                    cluster[j].append(i)
        for i in range(len(cluster)):
            size.append(len(cluster[i]))
        #print("cluster: ", cluster)
        return cluster, size


    def purity_score(self, y_true, y_pred):
        '''
        @topic: Evaluate the purity score.
        @parameters:
            1. y_true: ground truth label.
            2. y_pred: predicted label.
        '''
        # Compute confusion matrix and purity score
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


if __name__ == "__main__":
    # Pass parameters
    #dataset_name = "Iris.csv"
    #k_val = 3
    dataset_name, k_val = pass_parameters()
    # Input dataset
    X, Y = input_dataset(dataset_name)
    # Initialize the model
    em = EM_Algorithm(X, k_val, init_rd=False)
    # Training and predicting
    mean, cov, predicted_label, num_iters = em.em_clustering()
    # Sort the mean based on its norm and return its index
    mean_norm = [np.linalg.norm(mean[i]) for i in range(len(mean))]
    mean_norm_sorted_idx = np.argsort(mean_norm)
    mean_sorted = mean[mean_norm_sorted_idx]
    cov_sorted = cov[mean_norm_sorted_idx]
    print("Part A: ")
    print("Mean: \n", np.around(mean_sorted, decimals=3))
    print("#"*30)
    print("Part B: ")
    print("Covariance: \n", np.around(cov_sorted, decimals=3))
    print("#"*30)
    #print("predicted_label: \n", predicted_label)
    print("Part C: ")
    print("Number of iterations: ", num_iters)
    print("#"*30)
    # Output the clusters and its size
    clusters, size = em.cluster_membership(predicted_label)
    clusters_sorted = np.array(clusters)[mean_norm_sorted_idx]
    size_sorted = np.array(size)[mean_norm_sorted_idx]
    print("Part D: ")
    print("Cluster Membership: \n")
    for i in range(len(clusters)):
        print("Cluster {0}: \n".format(i+1))
        print(clusters_sorted[i])
    print("#"*30)
    print("Part E: ")
    print("The size of clusters: ", size_sorted)
    print("#"*30)
    # Evaluation
    purity = em.purity_score(Y, predicted_label)
    print("Part F: ")
    print("The purity score is: ", np.around(purity, decimals=3))
    print("#"*30)