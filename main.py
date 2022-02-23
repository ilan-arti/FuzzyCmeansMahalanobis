from dataclasses import dataclass

from matplotlib import pyplot as plt
import numpy as np
import sklearn

@dataclass
class GaussianParams:
    prob: float
    mean: np.ndarray
    cov: np.ndarray


def run_example_1(number_of_clusters=3, total_number_of_examples = 999): #balanced classes
    diagonals_of_covariance = np.random.uniform(-2,2,(number_of_clusters, 2))
    covariances = [np.diag(diag) for diag in diagonals_of_covariance]
    for i in range(number_of_clusters):
        mat = covariances[i]
        mat[1,0] = np.random.uniform(-np.sqrt(mat[0, 0]* mat[1, 1]))

    classes = [GaussianParams(1/number_of_clusters,np.random.uniform(-5,5,2),covariances[n]) for n in range(number_of_clusters)]
    data = []
    for class_ in classes:
        data.append(np.random.multivariate_normal(class_.mean,class_.cov,int(class_.prob*total_number_of_examples)))

    data = np.vstack(data)
    plt.plot(data[:,0],data[:,1],"*")
    plt.show()
if __name__=="__main__":
    run_example_1()


