import numpy as np
from sklearn.cluster import DBSCAN

def clustering_by_dbscan(distance_matrix, eps=70):
        """
        :param eps: unit m for Frechet distance, m^2 for Area
        """
        cl = DBSCAN(eps=eps, min_samples=1, metric='precomputed')
        cl.fit(distance_matrix)
        return cl.labels_

if __name__ == "__main__":
    dis_mat = np.loadtxt("dismat_area.txt",delimiter=',')
    labels = clustering_by_dbscan(dis_mat)
    print(labels)
    f = open("labels_a.txt",'w')
    for i in range(len(labels)):
        f.write(str(labels[i]))
        if i<(len(labels) - 1):
            f.write(',')
    f.close()
    print("labels get")