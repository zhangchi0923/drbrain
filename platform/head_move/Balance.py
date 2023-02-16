import datetime
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from numpy import linalg as LA


class Balancer(object):

    def __init__(self, out_path, mode):
        self.out_path = out_path
        self.mode = mode

    def text2Df(self, myStr: str):
        lines = myStr.split('\n')
        head = lines[0].split(',')
        arr = [x.split(',') for x in lines[1:] if not not x]
        df = pd.DataFrame(arr, columns=head)
        df['timestamp'] = df['timestamp'].astype('uint64', errors='ignore')
        df['difficulty'] = df['difficulty'].astype(int, errors='ignore')
        df['pos_x'] = df['pos_x'].astype(float, errors='ignore')
        df['pos_y'] = df['pos_y'].astype(float, errors='ignore')
        df['pos_z'] = df['pos_z'].astype(float, errors='ignore')
        return df

    def draw_sav(self, h_data: pd.DataFrame, mode: str, fig_num: int, read_path:str, sav_path: str):
        assert mode in ['sit', 'stand', 'head'], "invalid mode!"
        assert fig_num == 1 or fig_num == 2 or fig_num == 3, "input fig_num not among [1, 2, 3]!"
        if mode == 'sit':
            if fig_num == 1:
                cord_label = ['pos_x', 'pos_z']
                ext = [-75, 75, -125, 25]
            elif fig_num == 2:
                cord_label = ['pos_y', 'pos_z']
                ext = [-100, 100, -165, 35]
            else:
                cord_label = ['pos_y', 'pos_x']
                ext = [-100, 100, -100, 100]
        elif mode == 'stand':
            if fig_num == 1:
                cord_label = ['pos_x', 'pos_z']
                ext = [-150, 150, -270, 30]
            elif fig_num == 2:
                cord_label = ['pos_y', 'pos_z']
                ext = [-140, 160, -270, 30]
            else:
                cord_label = ['pos_y', 'pos_x']
                ext = [-150, 150, -150, 150]
        else:
            if fig_num == 1:
                cord_label = ['pos_x', 'pos_z']
                ext = [-40, 40, -68, 12]
            elif fig_num == 2:
                cord_label = ['pos_y', 'pos_z']
                ext = [-40, 40, -65, 15]
            else:
                cord_label = ['pos_y', 'pos_x']
                ext = [-50, 50, -55, 45]
        

        points = np.array(h_data[cord_label])
        hull = ConvexHull(points, incremental=True)

        try:
            im = plt.imread(read_path + '{}-{}@2x.png'.format(mode, fig_num))
        except Exception as e:
            raise e
        
        plt.imshow(im, extent=ext)
        plt.plot(h_data[cord_label[0]], h_data[cord_label[1]], lw=0.2, alpha=0.5)
        new_vertices = np.concatenate((hull.vertices, [hull.vertices[0]]))
        plt.plot(points[new_vertices, 0], points[new_vertices, 1], 'g', lw=0.5, alpha=0.5)
        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color='grey', alpha=0.2)
        plt.axis('off')
        plt.savefig(sav_path, dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close()

        return

    def get_logger(self, log_id, pth):
        date_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        exe_logger = logging.getLogger()
        exe_logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler(os.path.join(pth, 'log_' + date_time + '_' + log_id))
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        exe_logger.addHandler(handler)
        return exe_logger

    def calc_vel(self, h_data: pd.DataFrame):
        pos_mat = np.array(h_data[['timestamp', 'pos_x', 'pos_y', 'pos_z']])
        dlt_pos_mat = np.abs(np.diff(pos_mat, axis=0))
        shp = dlt_pos_mat.shape
        new_dlt_pos_mat = dlt_pos_mat[shp[0] % 10 :].reshape((
            shp[0] // 10, 10, shp[-1]
        )).sum(axis=1)
        def div_by_time(x):
            return np.array([x[1]/x[0], x[2]/x[0], x[3]/x[0]])*10
        vel_mat = np.apply_along_axis(div_by_time, 1, new_dlt_pos_mat)
        vel_norm = LA.norm(vel_mat, ord=2, axis=1)

        return list(vel_norm)

    def train_time(self, b_data: pd.DataFrame):
        def label_area(row):
            if row['pos_x'] > 0 and row['pos_y'] > 0:
                return 1
            if row['pos_x'] < 0 and row['pos_y'] > 0:
                return 2
            if row['pos_x'] < 0 and row['pos_y'] < 0:
                return 3
            if row['pos_x'] > 0 and row['pos_y'] < 0:
                return 4

        area = b_data.apply(lambda row: label_area(row), axis=1)
        b_data['area'] = area
        res_df = pd.DataFrame([b_data.groupby('area')['status'].count(), b_data.groupby('area')['status'].sum()], index=['count', 'sum'], columns=[1, 2, 3, 4])
        res_df.loc['succeed'] = 1 - res_df.loc['sum'] / res_df.loc['count']
        res_df.loc['rate'] = res_df.loc['count'] / res_df.loc['count'].sum()
        res_df = res_df.fillna(0)
        return res_df