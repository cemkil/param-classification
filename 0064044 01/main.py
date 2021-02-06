
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def create_dataset():
    class_means = np.array([[+0.0, +2.5],
                            [-2.5, -2.0],
                            [+2.5, -2.0]])
    covariance_matrix = np.array([[[3.2, +0.0], [+0.0, +1.2]],
                                  [[+1.2, -0.8], [-0.8, +1.2]],
                                  [[+1.2, +0.8], [+0.8, +1.2]]])
    class_sizes = np.array([120, 90, 90])

    points1 = np.random.multivariate_normal(class_means[0, :], covariance_matrix[0, :, :], class_sizes[0])
    points2 = np.random.multivariate_normal(class_means[1, :], covariance_matrix[1, :, :], class_sizes[1])
    points3 = np.random.multivariate_normal(class_means[2, :], covariance_matrix[2, :, :], class_sizes[2])

    X = np.vstack((points1, points2, points3))
    y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))
    np.savetxt("lab01_data_set.csv", np.hstack((X, y[:, None])), fmt="%f,%f,%d")

    #plot data points generated
    plt.figure(figsize=(10, 10))
    plt.plot(points1[:, 0], points1[:, 1], "r.", markersize=10)
    plt.plot(points2[:, 0], points2[:, 1], "g.", markersize=10)
    plt.plot(points3[:, 0], points3[:, 1], "b.", markersize=10)

    plt.xlabel("x1")
    plt.ylabel("x2")

    return X, y, class_sizes


def estimate_parameters(X, y, class_sizes):
    K = np.max(y)
    means = np.array([[np.mean(X[y == (c + 1), d]) for c in range(K)]
                             for d in range(X.shape[1])])
    covariances = np.array([(np.dot(np.transpose(X[y == (c+1), :] - np.transpose(means[:,c])), X[y==(c+1),:] - np.transpose(means[:,c]) ) ) / class_sizes[c]
                                  for c in range(K)])

    class_priors = [np.mean(y == (c + 1)) for c in range(K)]

    print("\nmean values :")
    print(means)
    print("\ncovariance matrixes : ")
    print(covariances)
    print("\n class priors : " )
    print(class_priors)
    print()

    return means, covariances, class_priors

def calculate_scores(mean, covariance, prior, point):

    return ( -1/2*np.log(np.linalg.det(covariance))-1/2*np.dot(np.dot(np.transpose(point - mean), np.linalg.inv(covariance)), (point - mean))+np.log(prior))

def create_confusion_matrix(X, y, means, covariances, class_priors):
    predicted = np.zeros((y.size))
    i = 0
    for point in X:
        g1 = calculate_scores(means[:, 0],covariances[0,:,:], class_priors[0], point)
        g2 = calculate_scores(means[:, 1], covariances[1, :, :], class_priors[1], point)
        g3 = calculate_scores(means[:, 2], covariances[2, :, :], class_priors[2], point)
        if np.amax([g1, g2, g3]) == g1:
            predicted[i] = 1
        elif np.amax([g1, g2, g3]) == g2:
            predicted[i] = 2
        else:
            predicted[i] = 3
        i = i + 1
    confusion_matrix = pd.crosstab(predicted.astype(int), y, rownames = ['y_pred'], colnames = ['y_truth'])
    print(confusion_matrix)
    return predicted

def plot_decision_boundary(prediction , y, means, covariances, priors):

    x1_interval = np.linspace(-6, +6, 50)
    x2_interval = np.linspace(-6, +6, 50)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)

    discriminant_values = np.zeros((len(x1_interval), len(x2_interval), 3))
    for i, x2 in enumerate(x2_interval):
        for j, x1 in enumerate(x1_interval):
            for c in range(3):
                discriminant_values[i,j,c] = calculate_scores(means[:,c],covariances[c,:,:],priors[c], [x1,x2])

    A = discriminant_values[:, :, 0]
    B = discriminant_values[:, :, 1]
    C = discriminant_values[:, :, 2]

    A[(A < B) & (A < C)] = np.nan
    B[(B < A) & (B < C)] = np.nan
    C[(C < A) & (C < B)] = np.nan
    discriminant_values[:, :, 0] = A
    discriminant_values[:, :, 1] = B
    discriminant_values[:, :, 2] = C

    plt.figure(figsize=(10, 10))
    plt.plot(X[y == 1, 0], X[y == 1, 1], "r.", markersize=10)
    plt.plot(X[y == 2, 0], X[y == 2, 1], "g.", markersize=10)
    plt.plot(X[y == 3, 0], X[y == 3, 1], "b.", markersize=10)
    plt.plot(X[prediction != y, 0], X[prediction != y, 1], "ko", markersize=12, fillstyle="none")
    plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 1], levels=0, colors="k")
    plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 2], levels=0, colors="k")
    plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 1] - discriminant_values[:, :, 2], levels=0, colors="k")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

if __name__ == '__main__':
    X, y, class_sizes = create_dataset()
    means, covariances, class_priors = estimate_parameters(X, y, class_sizes)
    prediction = create_confusion_matrix(X, y, means, covariances, class_priors)
    plot_decision_boundary(prediction, y, means, covariances, class_priors )