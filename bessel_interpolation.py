import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

def getInterpolationPoints(controlPoints, tList):
    n = len(controlPoints)-1
    interPoints = []
    for t in tList:
        Bt = np.zeros(2, np.float64)
        for i in range(len(controlPoints)):
            comb = 1  
            if i == 1 or i == 2:
                comb = 3
            Bt = Bt + comb * np.power(1-t,n-i) * np.power(t,i) * np.array(controlPoints[i])
        interPoints.append(list(Bt))    
    return interPoints


def getControlPointList(pointsArray, k1=-1, k2=1):
	points = np.array(pointsArray)
	index = points.shape[0] - 2

	res = []
	for i in range(index):
		tmp = points[i:i+3]
		p1 = tmp[0]
		p2 = tmp[1]
		p3 = tmp[2]

		if k1 == -1:
			l1 = np.sqrt(np.sum((p1-p2)**2))
			l2 = np.sqrt(np.sum((p2-p3)**2))
			k1 = l1/(l1+l2)
			k2 = l2/(l1+l2)

		p01 = k1*p1 + (1-k1)*p2
		p02 = (1-k2)*p2 + k2*p3
		p00 = k2*p01 + (1-k2)*p02
		
		sub = p2 - p00
		p12 = p01 + sub
		p21 = p02 + sub

		res.append(p12)
		res.append(p21)
	pFirst = points[0] + 0.1*(res[0] - points[0])
	pEnd = points[-1] + 0.1*(res[-1] - points[-1])
	res.insert(0,pFirst)
	res.append(pEnd)

	return np.array(res)

if __name__ == "__main__":
    start_csv = 1
    end_csv = 9
    for i in range(start_csv,end_csv + 1,1):
        plt.figure(i)
        df = pd.read_csv("./rdp2_data/rdp_"+str(i)+".csv")
        data = np.array(df[['C_x','C_z']])
        
        dataT = data.T
        plt.subplot(1,2,1)
        plt.plot(dataT[0],dataT[1])

        points = data
        controlP = getControlPointList(points)
        l = len(points) - 1
        t = np.linspace(0,1,50)#两点之间插值个数
        plt.subplot(1,2,2)
        for j in range(l):
            p = np.array([points[j], controlP[2*j], controlP[2*j+1], points[j+1]])
            interPoints = getInterpolationPoints(p, t)
            x = np.array(interPoints)[:,0]
            y = np.array(interPoints)[:,1]
            plt.plot(x,y,color ='b')
        # plt.scatter(np.array(points)[:,0],np.array(points)[:,1],color='gray')


        
    plt.show()
    