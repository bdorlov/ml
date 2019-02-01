import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs

X_moons, y_moons = make_moons(n_samples=250, noise=0.1)
X_blobs, y_blobs = make_blobs(n_samples=200, centers=4)

'''plt.figure(figsize=(9, 6))
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons)

plt.figure(figsize=(9, 6))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs)

plt.show()'''

def dist(a, b):
    return np.linalg.norm(a - b)

X = X_blobs

N = 4

def step(y, centers):
    #points = np.array()
    for i, p in enumerate(X):
        dists = []
        min_center_index = -1
        min_d = -1
        for j, center in enumerate(centers):
            d = dist(p, center)
            if d < min_d or min_d < 0:
                min_d = d
                min_center_index = j
        y[i] = min_center_index

    for j, center in enumerate(centers):
        arr = np.array([X[i] for i in range(len(y)) if y[i] == j])
        centers[j] = np.sum(arr, axis=0) / arr.shape[0]
        

    return y, centers
        

def main():
    y = np.full(X.shape[0], np.nan)
    
    indexes = list(range(X.shape[0]))
    np.random.shuffle(indexes)
    center_indexes = np.array(indexes[0:N])
    centers = np.full((N, 2), np.nan)
    
    for j, index in enumerate(center_indexes):
        centers[j] = X[index, :]
        y[index] = j
        
    #print(centers)
    #print(y)
    
    while True:
        old_centers = centers.copy()
        y, centers = step(y, centers)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.scatter(centers[:, 0], centers[:, 1], c='b', s=140, marker='*')
        plt.pause(0.1)
        plt.clf()
        norm = np.linalg.norm(old_centers - centers)
        print(norm)
        if norm < 1e-8:
            print('break')
            break


    plt.show()


if __name__ == '__main__':
    main()
        
        



