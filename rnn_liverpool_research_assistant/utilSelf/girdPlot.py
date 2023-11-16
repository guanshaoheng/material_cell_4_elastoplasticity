import matplotlib.pyplot as plt
import os
import numpy as np


class gridPlot():
    """
    x = mydomain.getX()  # nodal coordinate
    bx = FunctionOnBoundary(mydomain).getX()

    # plot the model
    grid = gridPlot(nx=nx, ny=ny, x=x, savePath=loadInfor, bx=bx)
    grid.plot(numberFlag=True)
    """
    def __init__(self, nx, ny, x, savePath, gaussianPoints, bx=None):
        self.nx, self.ny, self.x = nx, ny, x
        self.gaussianPoints, self.bx, self.savePath = gaussianPoints, bx, savePath

    def plot(self, numberFlag=False, n=None, gaussianFlag=False, activaLearningFlag=False):
        x_check = self.x.toListOfTuples()
        if gaussianFlag:
            fig = plt.figure(figsize=(8, 13))
            gx = self.gaussianPoints.toListOfTuples()# used to collect the nodes on the shear band
            if len(gx) == 32:
                ABCDnodeNumber = [1, 22, 9, 30]    # 2-4
            elif len(gx) == 128:
                ABCDnodeNumber = [1, 78, 48, 111]  # 4-8
            else:
                ABCDnodeNumber = [64, 351, 193, 479]  # 8-16
            keyCoord1 = [gx[i] for i in ABCDnodeNumber[:2]]
            keyCoord2 = [gx[i] for i in ABCDnodeNumber[2:]]
            LINE1 = lambda x: (keyCoord1[1][1]-keyCoord1[0][1])/(keyCoord1[1][0]-keyCoord1[0][0])*(x-keyCoord1[0][0])+keyCoord1[0][1]
            LINE2 = lambda x: (keyCoord2[1][1]-keyCoord2[0][1])/(keyCoord2[1][0]-keyCoord2[0][0])*(x-keyCoord2[0][0])+keyCoord2[0][1]
            shearBand, upperTriangle, lowerTriangle = [], [], []
            for i, coor_g in enumerate(gx):
                temp1 = LINE1(coor_g[0])
                if coor_g[1] < temp1*0.9:  # lower triangle
                    lowerTriangle.append(i)
                    plt.plot(coor_g[0], coor_g[1], 'b.')
                else:
                    temp2 = LINE2(coor_g[0])
                    if coor_g[1] > temp2*1.1:  # upper triangle
                        upperTriangle.append(i)
                        plt.plot(coor_g[0], coor_g[1], 'b.')
                    else:
                        shearBand.append(i)
                        plt.scatter(coor_g[0], coor_g[1], color='r', marker='o', s=70)
                        # plt.scatter(coor_g[0], coor_g[1], color='r', marker='o', s=70) # 8-16
                # plt.text(coor_g[0], coor_g[1], str(i), fontsize=8)
            with open(os.path.join(self.savePath, 'gaussianPointsList.txt'), 'w') as f:
                f.write('Upper Triangle\n')
                f.write(' '.join([str(i) for i in upperTriangle])+'\n')
                f.write('\n')
                f.write('Lower Triangle\n')
                f.write(' '.join([str(i) for i in lowerTriangle])+'\n')
                f.write('\n')
                f.write('Shear Band\n')
                f.write(' '.join([str(i) for i in shearBand])+'\n')
                f.write('\n')
        elif activaLearningFlag:
            fig = plt.figure(figsize=(8, 10))
            f = open('./FEMxML/activeModels/samplingIndex.txt')
            datas = f.readlines()
            f.close()
            i = 0
            while True:
                if 'Index of argsort' in datas[i]:
                    i += 1
                    temp = datas[i][:-1].split(' ')
                    sampleIndex = np.array([int(temp[j]) for j in range(len(temp))])
                elif 'Variance of the prediction' in datas[i]:
                    i += 1
                    temp = datas[i].split(' ')
                    sampleVariance = np.array([float(temp[j]) for j in range(len(temp))])
                    break
                else:
                    i += 1
            gx = self.gaussianPoints.toListOfTuples()
            gx_array = np.array([np.array([gx[i][0], gx[i][1]]) for i in range(len(gx))])
            usedIndex = sampleIndex[:int(0.4*len(sampleIndex))]
            plt.scatter(gx_array[usedIndex, 0], gx_array[usedIndex, 1],
                c=sampleVariance[usedIndex], marker='o', s=100, cmap='rainbow')
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=30)
        else:
            fig = plt.figure(figsize=(4, 6.5))
        xx = [x_check[i][0] for i in range(len(x_check))]
        yy = [x_check[i][1] for i in range(len(x_check))]
        # plt.scatter(xx, yy, label='Node')
        if numberFlag:
            for i in range(len(xx)):
                plt.text(xx[i], yy[i], str(i))
        if self.bx:
            if not activaLearningFlag:
                bx_check = self.bx.toListOfTuples()
                bxx = [bx_check[i][0] for i in range(len(bx_check))]
                byy = [bx_check[i][1] for i in range(len(bx_check))]
                plt.scatter(bxx, byy, label='Boundary')
                if numberFlag:
                    for i in range(len(bxx)):
                        plt.text(bxx[i], byy[i], str(i))
        minx, maxx, miny, maxy = min(xx), max(xx), min(yy), max(yy)
        plt.plot([minx, maxx], [miny, miny], color='k')
        plt.plot([minx, minx], [miny, maxy], color='k')
        plt.plot([maxx, maxx], [miny, maxy], color='k')
        plt.plot([minx, maxx], [maxy, maxy], color='k')
        # plt.legend()
        plt.axis('equal')
        # ax.legend(
        #     loc='lower center',
        #           # bbox_to_anchor=(0, -0.01),
        #           fancybox='sawtooth', shadow=True,
        #            markerscale=0.6, ncol=2)
        # plt.xlim([-0.01, 0.06])
        plt.tight_layout()
        plt.show()
        # plt.savefig(os.path.join(self.savePath, 'plot_node_model%d.svg' % (n if n else -1)))
