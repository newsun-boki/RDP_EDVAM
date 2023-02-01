import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    color_list = ['b','c','g','k','m','r','y','w','gold']
    start_csv = 1
    end_csv = 63
    labels = np.loadtxt('labels_f.txt',delimiter=',')
    label = 0
    for i in range(start_csv,end_csv + 1,1):
        plt.figure(1)

        rdp_df = pd.read_csv("./rdp2_data/rdp_"+str(i)+".csv")
        rdp_data = np.array(rdp_df[['C_x','C_z']])
        if label == int(labels[i-1]):
            rdp_dataT = rdp_data.T
            plt.plot(rdp_dataT[0],rdp_dataT[1],color=color_list[int(label)])
            label = label + 1

    plt.show()
    