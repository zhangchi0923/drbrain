"""
 author: zhangchi
 email: zcsjtu0923@gmail.com
 create date: 2023-03-29 14:44:32
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

class Firefly:
    def __init__(self, df, savePath):
        self._df = df
        self._savePath = savePath
    
    def plot(self):
        x = self._df['x']
        y = self._df['y']
        # plot two figures separately and save them without white spaces
        with plt.rc_context():
            img = plt.imread('src/fire_fly_bg.png')
            plt.imshow(img, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
            plt.plot(x, y, color="red", linewidth=0.5)
            plt.axis('off')
            # save figure without white spaces
            plt.savefig(self._savePath+'/traj.png', bbox_inches='tight')
            plt.close()

            from scipy.stats.kde import gaussian_kde
            k = gaussian_kde(np.vstack([x, y]))
            xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
            zi = k(np.vstack([xi.flatten(), yi.flatten()]))

            plt.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.4, cmap='jet')
            plt.xlim(x.min(), x.max())
            plt.ylim(y.min(), y.max())
            im = plt.imread('src/fire_fly_bg.png')
            plt.imshow(im, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
            plt.axis('off')
            # save figure without white spaces
            plt.savefig(self._savePath+'/heatmap.png', bbox_inches='tight')
            plt.close()


    