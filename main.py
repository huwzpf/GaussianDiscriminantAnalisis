import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def prob_x_given_y(arg, mean, covariance):
    a = -0.5 * np.transpose(arg - mean).dot((np.linalg.inv(covariance)).dot(arg - mean))
    b = 1/(((2 * np.pi)**(covariance.shape[0]/2)) * np.sqrt(np.linalg.det(covariance)))

    return b * np.exp(a)


def predict(arg, mean_pos, mean_neg, covariance, prior):
    pos = prob_x_given_y(arg, mean_pos, covariance) * prior
    neg = prob_x_given_y(arg, mean_neg, covariance) * (1 - prior)
    if pos > neg:
        return 1
    else:
        return 0


def train(x, y):
    n = len(y)
    labels = y.reshape(n, 1)

    positive_mean_numerator = 0
    negative_mean_numerator = 0
    y_positive_cnt = 0
    y_negative_cnt = 0
    for i in range(n):
        if labels[i] == 1:
            positive_mean_numerator += x[i, :]
            y_positive_cnt += 1
        else:
            negative_mean_numerator += x[i, :]
            y_negative_cnt += 1
    mean_positive = np.array((positive_mean_numerator / y_positive_cnt))
    mean_negative = np.array((negative_mean_numerator / y_negative_cnt))
    class_prior = y_positive_cnt / n

    covariance = 0
    y_neg = k = np.array([0 if a == 1 else 1 for a in labels]).reshape(n, 1)
    temp = labels * (x - mean_positive) + y_neg * (x - mean_negative)
    covariance = (np.transpose(temp).dot(temp))/n
    for i in range(n):
        if labels[i] == 0:
            color = '#ff2200'
        else:
            color = '#1f77b4'
        plt.scatter(x[i, 0], x[i, 1], c=color)

    axes = plt.gca()
    (x_min, x_max) = axes.get_xlim()
    (y_min, y_max) = axes.get_ylim()
    # arbitrary number
    elements = n * 2
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, elements), np.linspace(y_min, y_max, elements))
    p = np.empty((elements, elements))
    for i in range(elements):
        for j in range(elements):
            k = np.array([x_grid[i, j], y_grid[i, j]])
            p[i, j] = predict(k.reshape(x.shape[1], 1), mean_positive.reshape(x.shape[1], 1),
                              mean_negative.reshape(x.shape[1], 1), covariance, class_prior)
    plt.contour(x_grid, y_grid, p, levels=[0.5])
    plt.show()



if __name__ == '__main__':
    data = pd.read_csv('data.csv').dropna()
    x = data.iloc[:, :-1].to_numpy()
    y = data.iloc[:, -1].to_numpy()
    train(x, y)
