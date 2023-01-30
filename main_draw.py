import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    start_csv = 1
    end_csv = 9
    for i in range(start_csv,end_csv + 1,1):
        plt.figure(i)
        df = pd.read_csv("./data/"+ str(i)+".csv")
        data = np.array(df[['C_x','C_z']])
        
        dataT = data.T
        plt.subplot(1,2,1)
        plt.plot(dataT[0],dataT[1])

        rdp_df = pd.read_csv("./rdp_data/rdp_"+str(i)+".csv")
        rdp_data = np.array(rdp_df[['C_x','C_z']])
        
        rdp_dataT = rdp_data.T
        plt.subplot(1,2,2)
        plt.plot(rdp_dataT[0],rdp_dataT[1])
    plt.show()
    