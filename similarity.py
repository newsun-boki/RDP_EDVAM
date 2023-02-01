import similaritymeasures
import numpy as np
from sklearn.cluster import DBSCAN
import pandas as pd
def compute_distance_matrix(trajectories, method="Frechet"):
        """
        :param method: "Frechet" or "Area"
        """
        n = len(trajectories)
        dist_m = np.zeros((n, n))
        for i in range(n - 1):
            p = trajectories[i]
            for j in range(i + 1, n):
                q = trajectories[j]
                if method == "Frechet":
                    dist_m[i, j] = similaritymeasures.frechet_dist(p, q)
                else:
                    dist_m[i, j] = similaritymeasures.area_between_two_curves(p, q)
                dist_m[j, i] = dist_m[i, j]
        return dist_m


if __name__ == "__main__":
    start_csv = 1
    end_csv = 63
    trajectories = []
    for i in range(start_csv,end_csv + 1,1):
        df = pd.read_csv("./rdp2_data/rdp_"+ str(i)+".csv")
        data = np.array(df[['C_x','C_z']])
        trajectories.append(data)

    dis_mat = compute_distance_matrix(trajectories)
    arr = np.array(dis_mat)
    np.savetxt("dismat_frechet.txt",dis_mat,fmt='%f',delimiter=',')
    print("distance matrix Frechet get")

    dis_mat_area = compute_distance_matrix(trajectories,'area')
    arr = np.array(dis_mat_area)
    np.savetxt("dismat_area.txt",dis_mat_area,fmt='%f',delimiter=',')
    print("distance matrix area get")


    