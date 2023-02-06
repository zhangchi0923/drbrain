# -*- coding: utf-8 -*-

"""
 author: zhangchi
 email: zcsjtu0923@gmail.com
 create date: 2022-12-15 11:36:19
"""

import datetime
import logging
import os
import sys
import requests
import json
import hashlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from numpy import linalg as LA

from flask import Flask, request, jsonify
app = Flask(__name__)

def text2Df(myStr: str):
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

def draw_sav(h_data: pd.DataFrame, mode: str, fig_num: int, read_path:str, sav_path: str):
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
    if len(hull.vertices) == 0:
        logger.error("No calculated hull vertices!")

    try:
        im = plt.imread(read_path + '{}-{}@2x.png'.format(mode, fig_num))
    except Exception:
        logger.error("Read background figure error!")
        # raise
        # sys.exit()

    plt.imshow(im, extent=ext)
    plt.plot(h_data[cord_label[0]], h_data[cord_label[1]], lw=0.2, alpha=0.5)
    new_vertices = np.concatenate((hull.vertices, [hull.vertices[0]]))
    plt.plot(points[new_vertices, 0], points[new_vertices, 1], 'g', lw=0.5, alpha=0.5)
    plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color='grey', alpha=0.2)
    plt.axis('off')
    plt.savefig(sav_path, dpi=400, bbox_inches='tight', pad_inches=0)
    plt.close()

    return

def get_logger(log_id, pth):
    date_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    exe_logger = logging.getLogger()
    exe_logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(os.path.join(pth, 'log_' + date_time + '_' + log_id))
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    exe_logger.addHandler(handler)
    return exe_logger

def mkdir_new(path):
    if not os.path.exists(path):
        os.makedirs(path) 

def calc_vel(h_data: pd.DataFrame):
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

def train_time(b_data: pd.DataFrame):
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

from settings import SALT
def get_md5(d):
    d = dict(sorted(d.items()))
    d = SALT + str(d)
    md5 = hashlib.md5()
    md5.update(d.encode('utf-8'))
    return md5.hexdigest()
    
@app.route('/balance', methods=['POST'])
def balance():
    args = request.get_json(force=True)
    auth = request.headers.get('Authorization')
    auth_srv = get_md5(args)
    if auth_srv != auth:
        logger.error('Authorization failed.\nAuth from client:{}\nAuth from server:{}'.format(auth, auth_srv))
        return jsonify({
            'code':401,
            'msg':'Authorization failed.',
            'body':None
        })
    url_h = args['urlHead']
    url_b = args['urlBall']
    in_path = args['inPath']
    out_path = args['outPath']
    mode = args['mode']

    try:
        # log setting
        tmp, sid = os.path.split(url_h)
        # print(sid)
        mkdir_new(out_path + '/log')
        logger = get_logger(sid, out_path + '/log')
        logger.info('Authorization succeed.')
        # requests to get data
        url_data = requests.get(url_h)
        if url_data.status_code != 200:
            logger.error("Cannot access url data!")
            sys.exit()
        txt_head_data = text2Df(url_data.text)
        url_data.close()
        des_head_data = txt_head_data.describe()

        ball_data = pd.read_csv(url_b)
        train_res = train_time(ball_data)

        mkdir_new(out_path + '/result_fig')
        for n in [1, 2, 3]:
            draw_sav(txt_head_data, mode, n, in_path+'/src_fig/', out_path+'/result_fig/'+"traj_{}_{}.png".format(mode, n))
            logger.info("traj_{}_{} completed.".format(mode, n))
        vel_list = calc_vel(txt_head_data)

        out_dict = {'code':200,
                    'msg':'Computation completed successfully.',
                    'body':{
                        'forth': '{:.2f}'.format(des_head_data.loc['max', 'pos_y']), 
                        'back': '{:.2f}'.format(abs(des_head_data.loc['min', 'pos_y'])), 
                        'left': '{:.2f}'.format(des_head_data.loc['max', 'pos_x']), 
                        'right': '{:.2f}'.format(abs(des_head_data.loc['min', 'pos_x'])), 
                        'up': '{:.2f}'.format(des_head_data.loc['max', 'pos_z']), 
                        'down': '{:.2f}'.format(abs(des_head_data.loc['min', 'pos_z'])),
                        
                        'rightForthSuc': '{:.2f}%'.format(train_res.loc['succeed', 1]*100), 
                        'leftForthSuc': '{:.2f}%'.format(train_res.loc['succeed', 2]*100), 
                        'leftBackSuc': '{:.2f}%'.format(train_res.loc['succeed', 3]*100),
                        'rightBackSuc': '{:.2f}%'.format(train_res.loc['succeed', 4]*100),
                        
                        'rightForthPct': '{:.2f}%'.format(train_res.loc['rate', 1]*100), 
                        'leftForthPct': '{:.2f}%'.format(train_res.loc['rate', 2]*100), 
                        'leftBackPct': '{:.2f}%'.format(train_res.loc['rate', 3]*100),
                        'rightBackPct': '{:.2f}%'.format(train_res.loc['rate', 4]*100),

                        'velocity':vel_list
                    }          
        }

        logger.info("Velocity calculated and saved.")
        logger.info("--------------------------------")
        logger.info("Forth: {:.2f}".format(des_head_data.loc['max', 'pos_y']))
        logger.info("Back: {:.2f}".format(des_head_data.loc['min', 'pos_y']))
        logger.info("Left: {:.2f}".format(des_head_data.loc['max', 'pos_x']))
        logger.info("Right: {:.2f}".format(des_head_data.loc['min', 'pos_x']))
        logger.info("Up: {:.2f}".format(des_head_data.loc['max', 'pos_z']))
        logger.info("Down: {:.2f}".format(des_head_data.loc['min', 'pos_z']))
        logger.info("--------------------------------")
        logger.info("Succeed rates calculated.")
        logger.info("Right_forth: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 1]*100, train_res.loc['rate', 1]*100))
        logger.info("Left_forth: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 2]*100, train_res.loc['rate', 2]*100))
        logger.info("Left_back: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 3]*100, train_res.loc['rate', 3]*100))
        logger.info("Right_back: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 4]*100, train_res.loc['rate', 4]*100))
        logger.info("--------------END---------------")

        return jsonify(out_dict)
    except Exception as e:
        logger.exception(e)
        return jsonify({'code':500, 'msg': str(e), 'body':None})

if __name__ == '__main__':
    app.run(port=6666, debug=True)
    
