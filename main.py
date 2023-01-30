import numpy as np
import pandas as pd
from rdp import rdp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    start_csv = 1
    end_csv = 9
    for i in range(start_csv,end_csv + 1,1):
        df = pd.read_csv("./data/"+ str(i)+".csv")
        data = np.array(df[['C_x','C_z']])
        
        # dataT = data.T
        # fig, (ax1,ax2) = plt.subplots(1,2)
        # ax1.plot(dataT[0],dataT[1])

        rdp_mask = rdp(data,algo="iter", return_mask=True)
        rdp_data = data[rdp_mask]
        
        # rdp_dataT = rdp_data.T
        # ax2.plot(rdp_dataT[0],rdp_dataT[1])
        # plt.show()
        drop_index = []
        for j in range(rdp_mask.shape[0]):
            if rdp_mask[j] == False:
                drop_index.append(j)
        rdp_df = df.drop(drop_index)
        rdp_df.to_csv("./rdp_data/rdp_"+str(i)+".csv",index = None)
        print("finish: "+str(i))