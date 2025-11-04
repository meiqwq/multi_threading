import numpy as np
from threading import Thread
from queue import Queue

def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

def assign_clusters(data, centers, labels, start, end):
    for i in range(start, end):
        labels[i] = np.argmin([euclidean_distance(data[i], c) for c in centers])

def update_centers(data, labels, k):
    new_centers = []
    for i in range(k):
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            new_centers.append(cluster_points.mean(axis=0))
        else:
            new_centers.append(data[np.random.randint(0, len(data))])
    return np.array(new_centers)

def kmeans_threading(data, k=3, max_iter=100, n_threads=4):
    n_samples = data.shape[0]
    centers = data[np.random.choice(n_samples, k, replace=False)]
    labels = np.zeros(n_samples, dtype=int)

    for it in range(max_iter):
        threads = []
        chunk = n_samples // n_threads
        for t in range(n_threads):
            start = t * chunk
            end = n_samples if t == n_threads - 1 else (t + 1) * chunk
            thread = Thread(target=assign_clusters, args=(data, centers, labels, start, end))
            threads.append(thread)
            thread.start()
        for t in threads:
            t.join()
        
        new_centers = update_centers(data, labels, k)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers


if __name__ == "__main__":
    np.random.seed(42)
    data = np.vstack([
        np.random.randn(100, 2) + [0, 0],
        np.random.randn(100, 2) + [5, 5],
        np.random.randn(100, 2) + [0, 5]
    ])
    labels, centers = kmeans_threading(data, k=3, max_iter=20, n_threads=4)
    print("聚类中心:\n", centers)
